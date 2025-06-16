#necessary imports
from fastapi import FastAPI, Query, Request
from fastapi.responses import RedirectResponse, JSONResponse
from dotenv import load_dotenv
import os, requests, urllib.parse, json
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import shutil
from query_handler import (
    build_query_sentence,
    embed_query_sentence,
    tokenize_fn,
    search_topk
)
from normalizers import normalize_type
import faiss, pickle, numpy as np
from response import generate_final_response

load_dotenv()
app = FastAPI()

# env vars
google_env = lambda k: os.getenv(k)
GOOGLE_CLIENT_ID = google_env("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = google_env("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = google_env("GOOGLE_REDIRECT_URI")
SCOPE = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "openid",
    "email"
]

# The login for the app
@app.get("/auth/login")
def login():
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPE),
        "access_type": "offline",
        "prompt": "consent"
    }
    url = f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"
    return RedirectResponse(url)

# Front-end base URL
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8501/")

@app.get("/auth/callback")
def callback(request: Request):
    """
    Google redirects here after user consent.
    1. Exchange the authorization code for access / refresh tokens
    2. Verify ID-token ➜ extract user_id
    3. Persist tokens in user_data/<user_id>/tokens.json
    4. Redirect the user’s browser back to Streamlit with ?user_id=...
       (Streamlit reads it via st.experimental_get_query_params)
    """
    code = request.query_params.get("code")
    if not code:
        return JSONResponse({"error": "No code found"}, status_code=400)

    # exchanging code and token
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    token_resp = requests.post(token_url, data=data).json()
    if "error" in token_resp:
        return JSONResponse(token_resp, status_code=400)

    access_token  = token_resp.get("access_token")
    refresh_token = token_resp.get("refresh_token")
    id_token_jwt  = token_resp.get("id_token")

    # verifying ID token, and state persist
    try:
        idinfo = id_token.verify_oauth2_token(
            id_token_jwt, google_requests.Request(), GOOGLE_CLIENT_ID
        )
        user_id    = idinfo["sub"]
        user_email = idinfo.get("email", "unknown")

        os.makedirs(f"user_data/{user_id}", exist_ok=True)
        with open(f"user_data/{user_id}/tokens.json", "w") as f:
            json.dump(
                {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "email": user_email,
                },
                f,
            )

    except ValueError as e:
        return JSONResponse(
            {"error": "Invalid ID token", "details": str(e)},
            status_code=400,
        )

    clear_downloads(user_id=user_id)
    # Back to Streamlit
    redirect_url = f"{FRONTEND_URL}?user_id={user_id}"
    return RedirectResponse(url=redirect_url, status_code=302)
    

#Loading the necessary files, we skip if we already have the files
@app.get("/drive/load_files")
def load_drive_files(user_id: str, force: bool = Query(False, description="Reload even if metadata exists")):
    """
    Step 1 of indexing: pull Drive file list and cache it.
    If `force=false` and drive_files.json already exists ⇒ skip.
    """
    meta_path = f"user_data/{user_id}/drive_files.json"

    # fast-exit
    if os.path.exists(meta_path) and not force:
        return {"message": "Metadata already exists – skipping. Force it to reload if you changed your files"}

    if force:
        clear_downloads(user_id)  # 🔥 Clear downloaded files on force reload

    # checking for tokens and access
    tok_path = f"user_data/{user_id}/tokens.json"
    if not os.path.exists(tok_path):
        return JSONResponse({"error": "User not authenticated."}, status_code=401)
    access_token = json.load(open(tok_path))["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    # pull all the files
    files, page_token = [], None
    while True:
        r = requests.get(
            "https://www.googleapis.com/drive/v3/files",
            headers=headers,
            params={
                "pageSize": 1000,
                "fields": (
                "nextPageToken,"
                "files("
                    "id,name,mimeType,modifiedTime,parents,"
                    "webViewLink,webContentLink,thumbnailLink"
                ")"
            ),
                "pageToken": page_token,
            },
        )
        if r.status_code != 200:
            return JSONResponse({"error": r.text}, status_code=400)
        payload = r.json()
        files.extend(payload.get("files", []))
        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    os.makedirs(f"user_data/{user_id}", exist_ok=True)
    json.dump(files, open(meta_path, "w"), indent=2)
    return {"message": f"Saved metadata for {len(files)} files."}

def clear_downloads(user_id: str):
    downloads_path = os.path.join("user_data", user_id, "downloads")
    if os.path.exists(downloads_path):
        shutil.rmtree(downloads_path)
        print(f"🧹 Cleared old downloads for {user_id}")
    os.makedirs(downloads_path, exist_ok=True)
    
def clear_user_cache(user_id: str):
    user_dir = f"user_data/{user_id}/downloads"
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
        print(f"🧹 Cleared download cache for {user_id}")
    
#indexing the metadata into vector + inverted
@app.get("/drive/index_metadata")
def index_metadata(user_id: str, force: bool = Query(False, description="Rebuild even if index exists")):
    base = f"user_data/{user_id}"
    idx_path = f"{base}/metadata.index"
    emb_path = f"{base}/embeddings.npy"
    map_path = f"{base}/metadata_mapping.pkl"
    inv_path = f"{base}/inverted_index.pkl"

    # fast exit
    if not force and all(os.path.exists(p) for p in (idx_path, emb_path, map_path, inv_path)):
        return {"message": "✅ Index already exists – skipping. Force it to reload if you changed your files"}

    # Load the metadata
    metadata_path = f"{base}/drive_files.json"
    if not os.path.exists(metadata_path):
        return JSONResponse({"error": "No metadata found. Run /drive/load_files first."}, status_code=400)

    with open(metadata_path, "r") as f:
        drive_files = json.load(f)

    # building the mapping and canonical templates
    mapping, templates = [], []
    for f in drive_files:
        name  = f.get("name")
        ftype = normalize_type(f.get("mimeType", ""))
        date  = f.get("modifiedTime", "")[:10]
        tokens = tokenize_fn(name or "")
        mapping.append({
             "id":   f["id"],
             "name": name,
             "type": ftype,
             "date": date,
             "link": f.get("webViewLink"),
             "raw" : f 
         })
        templates.append(build_query_sentence(name, ftype, date, list(tokens)))

    # embeddin all templates
    embs = np.stack([embed_query_sentence(t) for t in templates])

    # saving FAISS index
    os.makedirs(base, exist_ok=True)
    idx = faiss.IndexFlatL2(embs.shape[1])
    idx.add(embs)
    faiss.write_index(idx, idx_path)

    # persisisting artefacts
    np.save(emb_path, embs)
    with open(map_path, "wb") as f:
        pickle.dump(mapping, f)

    inverted = {}
    for i, rec in enumerate(mapping):
        for tok in tokenize_fn(rec["name"] or ""):
            inverted.setdefault(tok, []).append(i)
    with open(inv_path, "wb") as f:
        pickle.dump(inverted, f)

    return {"message": f"Indexed {len(mapping)} files: built vector & inverted index."}

#Handle the query
@app.post("/query")
async def query_endpoint(payload: dict):
    user_id  = payload.get("user_id")
    qtxt     = payload.get("query")
    history  = payload.get("history", [])

    if not user_id or not qtxt:
        return JSONResponse({"error":"user_id and query required"}, status_code=400)

    # search
    results = search_topk(user_id, qtxt, top_k=5)

    tok_path = f"user_data/{user_id}/tokens.json"
    if not os.path.exists(tok_path):
        return JSONResponse({"error":"User not authenticated."}, status_code=401)
    access_token = json.load(open(tok_path))["access_token"]
    
    # Final response generation
    return generate_final_response(qtxt, user_id, results, access_token, history)

#-------------------------------------TESTING-------------------------------------------
# def test_embed_sentences(user_id: str, num_samples: int = 5):
#     """
#     Load saved metadata and test the query sentences that are embedded during indexing.
#     Prints the sentence and shape of the embedding.
#     """
#     metadata_path = f"user_data/{user_id}/drive_files.json"
#     if not os.path.exists(metadata_path):
#         print("❌ No metadata file found. Run /drive/load_files first.")
#         return

#     with open(metadata_path, "r") as f:
#         drive_files = json.load(f)

#     print(f"\n🔍 Testing {min(num_samples, len(drive_files))} embedded metadata sentences:\n")

#     for i, f in enumerate(drive_files[:num_samples]):
#         name = f.get("name")
#         ftype = normalize_type(f.get("mimeType", ""))
#         date = f.get("modifiedTime", "")[:10]
#         tokens = tokenize_fn(name or "")
#         sentence = build_query_sentence(name, ftype, date, list(tokens))
#         embedding = embed_query_sentence(sentence)
#         print(f"📄 Sentence {i+1}:\n{sentence}")
#         print(f"📐 Embedding shape: {embedding.shape}\n{'-'*50}")
        
# if __name__ == "__main__":
#     # Simulate frontend interaction
#     user_id = "110913943088152059091"  # 🔁 Replace as needed
    # queries = [
    #     "Open my resume, it is in pdf",
    #     "Can you show me the folder for ophthmate?",
    #     "I want to understand DR in India"
    # ]

    # for q in queries:
    #     print("\n==============================")
    #     print(f"🧠 Running Query: {q}")
    #     results = process_user_query(user_id, q)
    #     print("🎯 Final Search Results:\n", results)
