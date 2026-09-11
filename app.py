import cv2
import re
import fitz
import numpy as np
import pandas as pd
import pytesseract
import json
import base64

# -------------------------------------------------
# Tesseract Path
# -------------------------------------------------

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# -------------------------------------------------
# Extract Fields
# -------------------------------------------------

def clean_epic_candidate(raw):
    """
    Normalize common OCR misreads inside an EPIC-ID-shaped token
    (3 letters + 7 digits), e.g. O<->0, I/l<->1, S<->5, B<->8.
    """
    raw = raw.strip().upper()
    if len(raw) < 10:
        return None

    letters = raw[:3]
    digits = raw[3:10]

    # Fix letters that OCR may have misread as digits
    letter_fix = {"0": "O", "1": "I", "5": "S", "8": "B"}
    letters = "".join(letter_fix.get(c, c) for c in letters)

    # Fix digits that OCR may have misread as letters
    digit_fix = {"O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"}
    digits = "".join(digit_fix.get(c, c) for c in digits)

    if letters.isalpha() and digits.isdigit():
        return letters + digits
    return None


def extract_epic_id(text):
    """
    Search text for an EPIC-ID-shaped token (3 letters + 7 digits).

    Tries three strategies, from strict to fuzzy, and returns the first hit:
      1. Strict pattern on the raw text (handles the common, clean case).
      2. Same pattern on a whitespace-stripped version of the text, so an
         ID split across a line break/stray space by OCR still matches.
      3. A brute-force sliding window over every 10-character run in the
         whitespace-stripped text, normalizing each window's letter/digit
         positions and accepting it if it *could* be a valid EPIC ID. This
         catches cases where OCR garbled a character outside the normal
         O/I/L/S/B/Z/G confusion set, or picked up a bit of watermark noise.
    """
    text = text.upper().replace("|", " ")

    pattern = r'[A-Z]{3}\s?[0-9OILSBZG]{7}'

    # Strategy 1: strict pattern, raw text
    for m in re.finditer(pattern, text):
        candidate = clean_epic_candidate(m.group().replace(" ", ""))
        if candidate:
            return candidate

    # Strategy 2: strict pattern, whitespace/punctuation stripped
    compact_text = re.sub(r'[^A-Z0-9]', '', text)
    for m in re.finditer(r'[A-Z]{3}[0-9OILSBZG]{7}', compact_text):
        candidate = clean_epic_candidate(m.group())
        if candidate:
            return candidate

    # Strategy 3: fuzzy sliding window over every 10-char run
    for i in range(len(compact_text) - 9):
        window = compact_text[i:i + 10]
        candidate = clean_epic_candidate(window)
        if candidate:
            return candidate

    return ""


def extract_fields(text, epic_text=""):

    row = {
        "serial_no": "",
        "epic_id": "",
        "name": "",
        "relative_name": "",
        "house_no": "",
        "age": "",
        "gender": ""
    }

    text = text.replace("|", " ")

    # EPIC ID -- check the dedicated top-strip OCR pass first (cleanest),
    # then fall back to searching the full card text.
    row["epic_id"] = extract_epic_id(epic_text) or extract_epic_id(text)

    # Serial Number
    m = re.search(r'^\s*(\d{1,5})', text)
    if m:
        row["serial_no"] = m.group(1)

    # Name
    m = re.search(
        r'Name\s*:\s*(.+)',
        text,
        re.IGNORECASE
    )
    if m:
        row["name"] = m.group(1).split("\n")[0].strip()

    # Relative Name
    m = re.search(
        r'(?:Father|Mother|Husband)s?\s*Name\s*:\s*(.+)',
        text,
        re.IGNORECASE
    )
    if m:
        row["relative_name"] = m.group(1).split("\n")[0].strip()

    # House Number
    m = re.search(
        r'House\s*Number\s*:\s*(.+)',
        text,
        re.IGNORECASE
    )
    if m:
        row["house_no"] = m.group(1).split("\n")[0].strip()

    # Age
    m = re.search(
        r'Age\s*:\s*(\d+)',
        text,
        re.IGNORECASE
    )
    if m:
        row["age"] = m.group(1)

    # Gender
    m = re.search(
        r'Gender\s*:\s*(Male|Female|Other)',
        text,
        re.IGNORECASE
    )
    if m:
        row["gender"] = m.group(1)

    return row


# -------------------------------------------------
# Detect Cards
# -------------------------------------------------

def detect_cards(page):

    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(
        gray,
        220,
        255,
        cv2.THRESH_BINARY_INV
    )[1]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cards = []

    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)

        area = w * h

        if (
            area > 100000
            and w > 250
            and h > 150
        ):
            cards.append((x, y, w, h))

    cards = sorted(
        cards,
        key=lambda b: (b[1], b[0])
    )

    return cards


# -------------------------------------------------
# OCR Card
# -------------------------------------------------

def ocr_card(card):
    """
    OCR the whole card for name/relative/house/age/gender, but instead of
    slicing off the entire right-hand column (which also removes the
    EPIC ID sitting in the top strip), we white-out only the photo
    rectangle itself. This keeps the EPIC ID text intact.
    """

    h, w = card.shape[:2]

    masked = card.copy()

    # Photo box only occupies the lower ~80% of the card height,
    # not the very top strip where the EPIC ID is printed.
    y1, y2 = int(h * 0.18), int(h * 0.98)
    x1, x2 = int(w * 0.70), int(w * 0.99)
    masked[y1:y2, x1:x2] = 255

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.threshold(
        gray,
        180,
        255,
        cv2.THRESH_BINARY
    )[1]

    text = pytesseract.image_to_string(
        gray,
        config="--psm 6"
    )

    return text


def ocr_epic_strip(card):
    """
    Dedicated, higher-zoom OCR pass on just the top strip of the card
    (serial-number box + EPIC ID). Small text, so we throw extra
    resolution and a couple of different preprocessing/PSM combos at it
    and merge all the resulting text -- cards with faint print or the
    diagonal "DELETED"/modification watermark respond differently to
    fixed vs. adaptive thresholding, so trying both raises the odds that
    at least one pass reads cleanly.
    """

    h, w = card.shape[:2]

    # A little taller than the strictly-needed area, to be safe on cards
    # with slightly taller header rows (e.g. the "S", "Q", "#2" modification
    # prefixes next to the serial number).
    top_strip = card[0:int(h * 0.18), :]

    gray = cv2.cvtColor(top_strip, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.medianBlur(gray, 3)

    fixed_thresh = cv2.threshold(
        gray,
        180,
        255,
        cv2.THRESH_BINARY
    )[1]

    otsu_thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    combined_text = ""

    for image in (fixed_thresh, otsu_thresh):
        for psm in ("7", "8"):
            combined_text += "\n" + pytesseract.image_to_string(
                image,
                config=f"--psm {psm}"
            )

    return combined_text


def extract_photo_base64(card):

    h, w = card.shape[:2]

    # Right side contains photo
    photo = card[
        int(h * 0.15):int(h * 0.95),
        int(w * 0.72):int(w * 0.98)
    ]

    success, buffer = cv2.imencode(".jpg", photo)

    if not success:
        return ""

    return base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")

# -------------------------------------------------
# Process PDF
# -------------------------------------------------

def process_pdf(pdf_path):

    rows = []

    doc = fitz.open(pdf_path)

    start_page = 3               # skip first 3 pages
    end_page = len(doc) - 2      # skip last 2 pages

    for page_no in range(start_page, end_page):

        print(f"\nProcessing Page {page_no + 1}")

        page = doc[page_no]

        pix = page.get_pixmap(
            matrix=fitz.Matrix(3, 3)
        )

        img = np.frombuffer(
            pix.samples,
            dtype=np.uint8
        ).reshape(
            pix.height,
            pix.width,
            pix.n
        )

        if pix.n == 4:
            page_img = cv2.cvtColor(
                img,
                cv2.COLOR_RGBA2BGR
            )
        else:
            page_img = cv2.cvtColor(
                img,
                cv2.COLOR_RGB2BGR
            )

        cards = detect_cards(page_img)

        debug = page_img.copy()

        for x, y, w, h in cards:
            cv2.rectangle(debug, (x, y), (x+w, y+h), (0,255,0), 3)

        #cv2.imwrite(f"debug_page_{page_no+1}.png", debug)

        print(f"Cards detected: {len(cards)}")

        for x, y, w, h in cards:

            card = page_img[y:y+h, x:x+w]

            text = ocr_card(card)
            epic_text = ocr_epic_strip(card)

            record = extract_fields(text, epic_text)

            if record["name"]:

                record["photo_base64"] = extract_photo_base64(card)

                rows.append(record)

    doc.close()

    # json format
    with open(
        "voter_data.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            rows,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("\nDone")
    print("Records extracted:", len(rows))
    print("Saved as voter_data.json")


# -------------------------------------------------
# Run
# -------------------------------------------------

process_pdf("8-1.pdf")
