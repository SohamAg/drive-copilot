# 📂 Drive Copilot — Google Drive Assistant (RAG‑based)

**Drive Copilot** is a local, privacy‑respecting AI assistant that connects to your Google Drive and answers questions about your files.
It combines semantic search (MiniLM + FAISS) with Retrieval‑Augmented Generation (OpenAI GPT) to locate the most relevant passages and return accurate, cited answers.

---

## 💡 Key Features

|                            |                                                                    |
| -------------------------- | ------------------------------------------------------------------ |
| 🔐 **Secure OAuth**        | Read‑only Google Drive access per user                             |
| 🔍 **Semantic search**     | Embeds file metadata with MiniLM and indexes in FAISS              |
| 📄 **Rich format support** | Google Docs, Sheets, Slides, PDFs, DOCX, XLSX, PPTX, CSV, …        |
| ✂️ **Smart extraction**    | Downloads / exports files, chunks text, embeds chunks, ranks top‑k |
| 🧠 **RAG answers**         | GPT generates answers using only retrieved context                 |
| 📎 **Source links**        | Every answer cites the underlying Drive files                      |

---

## 🛠 Tech Stack

* **FastAPI** – backend API & Google OAuth
* **Streamlit** – lightweight interactive UI
* **HuggingFace MiniLM** (`all-MiniLM-L6-v2`) – semantic embeddings
* **FAISS** – vector similarity search
* **OpenAI GPT** – answer generation
* **Google Drive API** – metadata search, download & export
* **PyMuPDF**, **python‑docx**, **openpyxl**, **python‑pptx**, **pandas** – file parsing

---

## 🚀 Setup Instructions

### 1 · Clone the repository

```bash
git clone https://github.com/yourname/drive-copilot.git
cd drive-copilot
```

### 2 · Create environment variables

Create **`backend/.env`** with:

```env
GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/callback

OPENAI_API_KEY=sk-...your-openai-key...

BACKEND_URL=http://localhost:8000
```

Required OAuth scopes:

```
https://www.googleapis.com/auth/drive.readonly
https://www.googleapis.com/auth/userinfo.email
```

### 3 · Install backend dependencies

```bash
cd backend
python -m venv venv            # optional but recommended
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4 · Start the FastAPI backend

```bash
uvicorn main:app --reload
```

Backend runs at **[http://localhost:8000](http://localhost:8000)**

### 5 · Run the Streamlit frontend

```bash
cd ../frontend
streamlit run streamlit_app.py
```

Opens a browser tab → log in with Google → start querying your Drive.

---

## 🧪 Example Prompts

```
You can ask it anything about a file/folder. Mentioning the name, or atleast something similar to the name, is imperative.
```

---

## 📁 Project Structure

```
drive-copilot/
├── backend/
│   ├── main.py            # FastAPI app
│   ├── response.py        # RAG pipeline & file handling
│   ├── query_handler.py   # embeddings & search
│   ├── search_metadata.py # FAISS + Drive helpers
│   ├── normalizers.py     # MIME-type helpers
│   └── user_data/         # per-user tokens, downloads, FAISS index
│
├── frontend/
│   └── streamlit_app.py   # Streamlit interface
├── requirements.txt
└── README.md
```

---

## ✅ Evaluation & Generalization

* Works on any Google Drive (personal or business)
* Supports native Google files **and** uploaded Office/PDFs
* Uses semantic retrieval — no overfitting to sample data

---

## 📄 License

MIT License  •  © 2024 Soham Agarwal
