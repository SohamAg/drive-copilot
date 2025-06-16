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

    # Google Native
    if mime == "application/vnd.google-apps.document": return "google_doc"
    if mime == "application/vnd.google-apps.spreadsheet": return "google_sheet"
    if mime == "application/vnd.google-apps.presentation": return "google_slide"
    if mime == "application/vnd.google-apps.folder": return "folder"
    if mime == "application/vnd.google-apps.form": return "form"
    if mime == "application/vnd.google-apps.drawing": return "drawing"
    if mime == "application/vnd.google-apps.script": return "script"

    # Office Formats
    if "spreadsheetml.sheet" in mime or mime.endswith("xlsx"): return "xlsx"
    if "presentationml.presentation" in mime or mime.endswith("pptx"): return "pptx"
    if "msword" in mime or "wordprocessingml.document" in mime or mime.endswith("docx"): return "docx"

    # Common formats
    if mime == "application/pdf": return "pdf"
    if "csv" in mime: return "csv"
    if "json" in mime: return "json"
    if mime.startswith("text/"): return "text"
    if "markdown" in mime: return "markdown"
    if "code" in mime or "python" in mime: return "code"
    if "epub" in mime or "mobi" in mime: return "ebook"
    if "zip" in mime or "compressed" in mime: return "archive"
    if "font" in mime: return "font"

    # Media
    if mime.startswith("image/"): return "image"
    if mime.startswith("video/"): return "video"
    if mime.startswith("audio/"): return "audio"

    # Fallbacks
    if mime.endswith(("png", "jpg", "jpeg", "webp")): return "image"
    if mime.endswith(("mp4", "mkv", "avi", "mov", "webm")): return "video"
    if mime.endswith(("mp3", "wav", "ogg")): return "audio"

    # Maha Fallback
    return mime.split("/")[-1]


