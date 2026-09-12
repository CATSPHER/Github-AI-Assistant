"""
All the knobs live here so the rest of the code never touches os.environ directly.
"""
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma_store")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GITHUB_PAT = os.getenv("GITHUB_PAT", "")

CHUNK_SIZE = 800         # characters per chunk — kept conservative so chunks stay
                          # under common embedding-model limits (e.g. all-MiniLM-L6-v2's
                          # 256-token cap); ~4 chars/token is a rough rule of thumb
CHUNK_OVERLAP = 150      # keeps context from being sliced mid-thought at chunk boundaries
TOP_K = 6                # how many chunks we retrieve per question

# Only index files that are actually useful for a code Q&A assistant.
ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rb", ".rs",
    ".md", ".mdx", ".txt", ".yaml", ".yml", ".json", ".toml",
}
# Common doc/config files that have no extension at all
ALLOWED_FILENAMES = {"README", "LICENSE", "Dockerfile", "Makefile", "CONTRIBUTING", "CHANGELOG"}
IGNORED_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build"}
MAX_FILE_SIZE_BYTES = 300_000  # skip huge generated/minified files