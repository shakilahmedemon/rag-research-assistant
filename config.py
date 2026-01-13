import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
PAPERS_DIR = os.path.join(BASE_DIR, "papers")
CHUNKS_FILE = os.path.join(MEMORY_DIR, "chunks.json")
EMBED_FILE = os.path.join(MEMORY_DIR, "chunk_embeddings.npy")

# Ensure directories exist
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(PAPERS_DIR, exist_ok=True)

# Retrieval
TOP_K = 5

# LLM
LLM_MODEL = "gemini-pro"  # Using Gemini Pro model
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1500

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200 # characters

# Logging
DEBUG = False  # Set to False for production

# Application metadata
APP_NAME = "Advanced RAG Academic Assistant"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Intelligent Research Paper Analysis & Question Answering System"
