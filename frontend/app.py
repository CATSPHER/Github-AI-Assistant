import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="AI GitHub Repo Assistant", page_icon="🔎")
st.title("🔎 AI GitHub Repository Assistant")

if "indexed_repo" not in st.session_state:
    st.session_state.indexed_repo = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def extract_error_detail(resp: requests.Response) -> str:
    """FastAPI puts the useful message in the JSON body's 'detail' field —
    resp.raise_for_status() alone only gives a generic reason phrase, so we
    pull the body out first."""
    try:
        return resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text or f"HTTP {resp.status_code}"


with st.sidebar:
    st.subheader("1. Index a repo")
    repo_url = st.text_input("GitHub repo URL", placeholder="https://github.com/owner/repo")
    if st.button("Ingest repo", type="primary", disabled=not repo_url):
        with st.spinner("Cloning, chunking, and embedding..."):
            try:
                resp = requests.post(f"{API_BASE}/ingest", json={"repo_url": repo_url}, timeout=600)
                if not resp.ok:
                    st.error(f"Ingest failed: {extract_error_detail(resp)}")
                else:
                    data = resp.json()
                    st.session_state.indexed_repo = repo_url
                    st.session_state.messages = []
                    st.success(f"Indexed {data['files']} files into {data['chunks']} chunks.")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the backend: {e}")

    if st.session_state.indexed_repo:
        st.caption(f"Currently indexed:\n{st.session_state.indexed_repo}")


st.subheader("2. Ask about it")
file_scope = st.text_input(
    "Scope to a file (optional)",
    placeholder="e.g. main.py or backend/rag_graph.py",
    key="file_scope",
)

if not st.session_state.indexed_repo:
    st.info("Index a repo in the sidebar first.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("How does authentication work in this project?")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/ask",
                        json={
                            "repo_url": st.session_state.indexed_repo,
                            "question": question,
                            "file_path": file_scope or None,
                        },timeout=120,
                    )
                    if not resp.ok:
                        st.error(f"Request failed: {extract_error_detail(resp)}")
                    else:
                        data = resp.json()
                        st.markdown(data["answer"])
                        if data.get("sources"):
                            st.caption("Sources: " + ", ".join(sorted(set(data["sources"]))))
                        st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not reach the backend: {e}")