import os
import json
import re
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from search_metadata import search_similar_metadata
from normalizers import normalize_extracted_type

load_dotenv()

# OpenAI setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

#query skeleton established
def query_openai(prompt: str, max_tokens: int = 150) -> str:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI error:", e)
        return ""

#query classification for later prompting, defunct logic
# def classify_query(query: str) -> str:
#     prompt = f"""
#     Classify the following user query into one of three categories:
#     1. File-based: Refers to or asks about a specific file.
#     2. Folder-based: Refers to or asks about a specific folder or its contents.
#     3. General: A broad query not tied to a file or folder.

#     Respond with only one of these words: File-based, Folder-based, or General.

#     Query: "{query}"
#     """
#     res = query_openai(prompt, max_tokens=20).lower()
#     if "file" in res: return "file"
#     if "folder" in res: return "folder"
#     if "general" in res: return "general"
#     return "general"

#extracting relavant metadata
def extract_metadata(query: str) -> dict:
    prompt = f"""
    Extract metadata from the following user query:
    - File or folder name (if any)
    - File type (like PDF, pptx (from presentation), spreadsheet (xlsx), image)
    - Date or time reference (month, year, etc.)

    Respond in JSON format with keys: name, type, date.
    Use null if a value is not found.

    Query: "{query}"
    """
    res = query_openai(prompt, max_tokens=100)
    # print("🧪 Metadata Response:", res)
    matches = re.findall(r'\{[^{}]+\}', res)
    for match in reversed(matches):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    return {"name": None, "type": None, "date": None}

#extracting relavant keywords
def extract_words(query: str) -> list[str]:
    prompt = f"""
    From the user query below, extract a list of the most meaningful keywords for document retrieval.
    - Include anything that is in double or single quotes.
    - Prioritize any file names, folder names, or project-specific terms.
    - Retain capitalized words (e.g., names) even if they are short.
    - Avoid common stopwords (like "the", "and", "to") and generic terms (like "file", "folder", "document").
    - Don't include months or years.

    Query: "{query}"

    Respond in this format: ["", "", ""]
    """
    res = query_openai(prompt, max_tokens=100)
    # print("🧪 Keywords Response:", res)
    matches = re.findall(r'\[[^\[\]]+\]', res)
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, list):
                return [x.strip() for x in parsed if isinstance(x, str)]
        except json.JSONDecodeError:
            continue
    return []

#tokenizing file names
def tokenize_fn(fn: str) -> set[str]:
    base = fn.rsplit('.', 1)[0]
    parts = re.split(r'[^A-Za-z0-9]+', base)
    toks = []
    for p in parts:
        toks += re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', p)
    STOPWORDS = {"in", "on", "at", "to", "of", "an", "is", "by", "as"}
    return {t.lower() for t in toks if len(t) > 1 and t.lower() not in STOPWORDS}

#building the query sentence to be embedded
def build_query_sentence(name, ftype, date, words):
    parts = [
        f"File: {name};" if name else "File: ;",
        f"Type: {ftype};" if ftype else "Type: ;",
        f"Modified: {date};" if date else "Modified: ;",
        f"Keywords: {', '.join(words)};" if words else "Keywords: ;"
    ]
    return " ".join(parts)

#embedding the query sentence
def embed_query_sentence(sentence: str) -> np.ndarray:
    # print(f'SENTENCE BEFORE EMBEDDING: {sentence}')
    emb = embedding_model.encode(sentence)
    return np.array(emb, dtype="float32")


# Main vector search function
def search_topk(user_id: str, query: str, top_k: int = 5):
    """
    Skip explicit query-classification. Always embed the query, keyword-filter,
    vector-search, return top-k metadata records.
    """
    # LLM extracts metadata & keywords
    meta      = extract_metadata(query)
    keywords  = extract_words(query)

    # Build canonical sentence & embed
    sentence  = build_query_sentence(
        meta.get("name"), normalize_extracted_type(meta.get("type")),
        meta.get("date"), keywords)
    q_emb     = embed_query_sentence(sentence)

    # Vector + keyword search
    return search_similar_metadata(user_id, q_emb, keywords, top_k, threshold=0.5)

#--------------------------------------TESTING-----------------------------------

#processing user query, now defunct only used for testing
# def process_user_query(user_id: str, query: str):
#     query_type = classify_query(query)
#     metadata = extract_metadata(query)
#     metadata["type"] = normalize_extracted_type(metadata.get("type"))
#     if query_type == "folder":
#         metadata["type"] = "folder"
#     keywords = extract_words(query)
#     sentence = build_query_sentence(metadata.get("name"), metadata.get("type"), metadata.get("date"), keywords)
#     embedding = embed_query_sentence(sentence)
#     results = search_similar_metadata(user_id, embedding, keywords, 5, 0.5)
#     print(f"🔍 Final Results for '{query_type}' query:\n", results)
#     return results

# processing searching and classifying, now defunct only used for testing
# def search_and_classify(
#         user_id: str, query: str,
#         k_file=2, k_folder=1, k_general=5):

#     q_type   = classify_query(query)

#     md       = extract_metadata(query)
#     md["type"] = normalize_extracted_type(md.get("type"))
#     if q_type == "folder":
#         md["type"] = "folder"

#     keywords = extract_words(query)
#     sent     = build_query_sentence(md.get("name"), md.get("type"), md.get("date"), keywords)
#     q_emb    = embed_query_sentence(sent)

#     k = {"file": k_file, "folder": k_folder, "general": k_general}[q_type]
#     results = search_similar_metadata(user_id, q_emb, keywords, k, 0.7)
#     return q_type, results

# if __name__ == "__main__":
#     test_queries = [
#         "Can you open harshit birthday video?",
#         # "Where is the OphthMate folder?",
#         # "I want information about diabetic retinopathy in India",
#         # "Can you show the image of our graduation ceremony?",
#         # "Find the CSV with my stock trades from March",
#         # "What did the PDF summary say about the client onboarding?"
#     ]

#     user_id = "110913943088152059091"  # Replace with actual user ID
#     for q in test_queries:
#         print("\n==============================")
#         print(f"🧠 Running Query: {q}")
#         results = search_topk(user_id, q)
#         print("→ Final Search Results:")
#         print("🔍 Top Results:\n", results if results else "No results found.")
