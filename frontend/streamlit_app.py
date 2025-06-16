import streamlit as st
import requests, webbrowser, os
import streamlit.components.v1 as components

#url
BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Drive Copilot", layout="centered")
st.title("🤖 Google Drive Copilot")

# session state management
if "user_id"   not in st.session_state: st.session_state.user_id   = None
if "meta_ok"   not in st.session_state: st.session_state.meta_ok   = False
if "index_ok"  not in st.session_state: st.session_state.index_ok  = False
if "history"   not in st.session_state: st.session_state.history   = []  # list[dict(q,a)]

# capturing user credentials
params = st.query_params
if "user_id" in params and not st.session_state.user_id:
    st.session_state.user_id = params["user_id"][0] if isinstance(params["user_id"], list) else params["user_id"]
    st.query_params.clear()        # tidy URL
    st.success("✅ Logged in via Google!")

# login logic
if not st.session_state.user_id:
    st.subheader("Connect your Google Drive")
    st.markdown("You must log in to load and search your Google Drive files.")
    if st.button("Login via Google OAuth"):
        webbrowser.open_new_tab(f"{BACKEND}/auth/login")
        st.info("A browser tab opened. Complete consent; you’ll return here automatically.")
    st.stop()

# sidebar logic
with st.sidebar:
    st.write(f"Logged in as the user: {st.session_state.user_id}")
    if st.button("🚪 Logout"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# Metadata + indexing logic
if not st.session_state.index_ok:
    st.subheader("Load your Drive files")
    st.markdown("Use the options below to load and index your Drive files. Force reload if new files were added.")

    # Metadata logic
    meta_ct = st.container()
    if not st.session_state.meta_ok:
        force_meta = meta_ct.checkbox("Force reload in case you added files", key="force_meta")
        if meta_ct.button("⬇️ Load Drive file metadata", key="load_meta_btn"):
            meta_ct.empty()  # hide UI during long call
            r = requests.get(f"{BACKEND}/drive/load_files",
                             params={"user_id": st.session_state.user_id, "force": force_meta})
            if r.status_code == 200:
                st.session_state.meta_ok = True
                st.success(r.json().get("message", "Metadata loaded."))
            else:
                st.error(r.json()); st.stop()

    # Indexing logic
    if st.session_state.meta_ok and not st.session_state.index_ok:
        idx_ct = st.container()
        force_idx = idx_ct.checkbox("force re-index in case you added files", key="force_idx")
        if idx_ct.button("⚙️ Build / Verify index", key="build_idx_btn"):
            idx_ct.empty()
            r = requests.get(f"{BACKEND}/drive/index_metadata",
                             params={"user_id": st.session_state.user_id, "force": force_idx})
            if r.status_code == 200 and any(k in r.json().get("message", "") for k in ("Indexed", "already exists")):
                st.session_state.index_ok = True
                st.success(r.json()["message"])
            else:
                st.error(r.json()); st.stop()

# chat ui logic
if st.session_state.index_ok:
    st.subheader("💬 Chat with your Drive")

    # Show history
    for h in st.session_state.history:
        with st.chat_message("user"):      st.markdown(h["q"])
        with st.chat_message("assistant"): st.markdown(h["a"])

    # Input
    user_q = st.chat_input("Ask about your files or folders…")
    if user_q:
        with st.chat_message("user"): st.markdown(user_q)
        answer_box = st.empty()            # placeholder to update once

        with st.spinner("Thinking…"):
            payload = {
                "user_id": st.session_state.user_id,
                "query":   user_q,
                "history": st.session_state.history[-5:],
            }
            r = requests.post(f"{BACKEND}/query", json=payload)

        if r.status_code == 200 and "answer" in r.json():
            data    = r.json()
            answer  = data["answer"]
            sources = data.get("sources", [])
            
            ICON = {
                "pdf": "📄", "google_doc": "📄", "text": "📄",
                "spreadsheet": "📊", "presentation": "📽️",
                "image": "🖼️", "video": "🎞️", "audio": "🎧",
                "folder": "📁"
            }
            
            with answer_box.container():
                st.markdown(answer)
                if sources:
                    st.markdown("**Sources**")
                    for s in sources:
                        ico   = ICON.get(s["type"], "📦")
                        link  = s.get("link")
                        thumb = s.get("thumb")

                        if s["type"] in ("image", "video"):
                            # Build the common iframe URL
                            preview_url = f"https://drive.google.com/file/d/{s['id']}/preview"
                            # Add autoplay only for videos
                            allow_attr = " allow=\"autoplay\"" if s["type"] == "video" else ""
                            # Define your dimensions (tweak as needed)
                            width, height = 640, 360
                            iframe = f"""
                            <iframe src="{preview_url}"
                                    width="{width}" height="{height}"{allow_attr}
                                    frameborder="0"></iframe>"""
                            # Streamlit will render it
                            components.html(iframe, height=height + 20)
                        else:
                            st.markdown(f"- {ico} [{s['name']}]({link})" if link else f"- {ico} {s['name']}")

            st.session_state.history.append({"q": user_q, "a": answer})
        else:
            answer_box.error(f"Backend error {r.status_code}: {r.text}")
