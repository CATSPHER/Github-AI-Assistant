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


st.divider()
st.subheader("3. Propose & apply a fix (writes to GitHub!)")

if "fix_pending" not in st.session_state:
    st.session_state.fix_pending = None

target_file = st.text_input("File to fix (exact path in repo)", placeholder="calculator.py")
fix_request = st.text_area("Describe the fix", placeholder="handle division by zero in the divide function")

if st.button("Propose fix", disabled=not (target_file and fix_request and st.session_state.indexed_repo)):
    with st.spinner("Fetching file and generating fix..."):
        try:
            resp = requests.post(
                f"{API_BASE}/fix/propose",
                json={"repo_url": st.session_state.indexed_repo, "target_file": target_file, "fix_request": fix_request},
                timeout=120,
            )
            if resp.ok:
                st.session_state.fix_pending = resp.json()
            else:
                st.error(f"Failed: {resp.json().get('detail', resp.text)}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach backend: {e}")

if st.session_state.fix_pending and st.session_state.fix_pending.get("status") == "awaiting_approval":
    pending = st.session_state.fix_pending
    if not pending.get("valid_syntax", True):
        st.warning("⚠️ Proposed content failed Python syntax validation — review carefully.")
    st.code(pending["diff"], language="diff")

    col1, col2 = st.columns(2)
    if col1.button("✅ Approve — create branch, push, open PR", type="primary"):
        with st.spinner("Creating branch, pushing files, opening PR..."):
            resp = requests.post(f"{API_BASE}/fix/approve", json={"thread_id": pending["thread_id"], "approved": True})
            result = resp.json()
            st.success(result.get("answer", ""))
            if result.get("pr_url"):
                st.markdown(f"[Open the PR]({result['pr_url']})")
            st.session_state.fix_pending = None
    if col2.button("❌ Reject"):
        resp = requests.post(f"{API_BASE}/fix/approve", json={"thread_id": pending["thread_id"], "approved": False})
        st.info(resp.json().get("answer", ""))
        st.session_state.fix_pending = None