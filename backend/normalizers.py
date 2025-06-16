TYPE_CANONICAL_MAP = {
    # Google Docs equivalents
    "word": "google_doc",
    "word doc": "google_doc",
    "doc": "google_doc",
    "docx": "google_doc",
    "google doc": "google_doc",
    "google docs": "google_doc",
    "text document": "google_doc",

    # Spreadsheets
    "excel": "spreadsheet",
    "xls": "spreadsheet",
    "xlsx": "spreadsheet",
    "sheet": "spreadsheet",
    "google sheet": "spreadsheet",

    # Presentations
    "powerpoint": "presentation",
    "ppt": "presentation",
    "pptx": "presentation",
    "google slides": "presentation",
    "slides": "presentation",

    # PDFs
    "pdf": "pdf",

    # Images
    "image": "image",
    "photo": "image",
    "picture": "image",
    "screenshot": "image",

    # Videos
    "video": "video",
    "clip": "video",

    # Folders
    "folder": "folder"
}

def normalize_extracted_type(user_type: str) -> str:
    if not user_type:
        return None
    return TYPE_CANONICAL_MAP.get(user_type.strip().lower(), user_type.strip().lower())

def normalize_type(mime: str) -> str:
    mime = mime.lower()

    # Native Google types
    if mime == "application/vnd.google-apps.spreadsheet":
        return "google_sheet"
    if mime == "application/vnd.google-apps.document":
        return "google_doc"
    if mime == "application/vnd.google-apps.presentation":
        return "google_slide"
    if mime == "application/vnd.google-apps.folder":
        return "folder"

    # Uploaded Office files
    if "spreadsheetml.sheet" in mime or mime.endswith("xlsx"):
        return "xlsx"
    if "presentationml.presentation" in mime or mime.endswith("pptx"):
        return "pptx"
    if "msword" in mime or mime.endswith("docx"):
        return "docx"

    # Generic fallbacks
    if "spreadsheet" in mime or "excel" in mime:
        return "spreadsheet"
    if "presentation" in mime or "powerpoint" in mime:
        return "presentation"
    if mime == "application/pdf":
        return "pdf"
    if mime.startswith("text/"):
        return "text"

    # Default: last part of the MIME string
    return mime.split("/")[-1]


