import json
import numpy as np
import os
from config import TOP_K, EMBEDDING_MODEL, CHUNKS_FILE, EMBED_FILE

def load_model():
    """Load the embedding model. This function will be used in the main app with caching."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL)
    except ImportError as e:
        raise ImportError(f"sentence_transformers package is not installed or has dependency issues. Error: {e}. Please install it using: pip install sentence-transformers")

def precompute_embeddings():
    """
    Computes embeddings for all chunks in chunks.json and saves to .npy
    Optimized to avoid re-computing if not needed could be added, 
    but for now we simply recompute all to ensure consistency.
    """
    if not os.path.exists(CHUNKS_FILE):
        return

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        # Save empty
        np.save(EMBED_FILE, np.array([]))
        return

    texts = [c["text"] for c in chunks]
    
    try:
        model = load_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        np.save(EMBED_FILE, embeddings)
        print(f"Computed embeddings for {len(chunks)} chunks.")
    except ImportError as e:
        print(f"Embedding computation skipped: {e}")
        # Create a dummy embeddings file if needed
        if len(chunks) > 0:
            # Create zero embeddings array with appropriate shape
            dummy_embeddings = np.zeros((len(chunks), 384))  # 384 is a common embedding size
            np.save(EMBED_FILE, dummy_embeddings)
            print(f"Created dummy embeddings file with shape {dummy_embeddings.shape}")
    except Exception as e:
        print(f"Error during embedding computation: {e}")
        # Still create a dummy embeddings file
        if len(chunks) > 0:
            dummy_embeddings = np.zeros((len(chunks), 384))
            np.save(EMBED_FILE, dummy_embeddings)
            print(f"Created dummy embeddings file due to error: {dummy_embeddings.shape}")

def retrieve_chunks(query, top_k=TOP_K):
    if not os.path.exists(CHUNKS_FILE):
        return []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        return []

    # Ensure embeddings exist
    if not os.path.exists(EMBED_FILE):
        precompute_embeddings()
        if not os.path.exists(EMBED_FILE):
            return []

    embeddings = np.load(EMBED_FILE)
    
    if len(embeddings) != len(chunks):
        # Mismatch detected, recompute
        precompute_embeddings()
        embeddings = np.load(EMBED_FILE)

    if len(embeddings) == 0:
        # Return chunks with dummy scores if we can't compute embeddings
        for chunk in chunks[:top_k]:
            chunk["score"] = 0.5  # dummy score
        return chunks[:top_k]

    try:
        model = load_model()
        query_embedding = model.encode([query], convert_to_numpy=True)[0]

        # Cosine similarity
        # Normalize embeddings for dot product to be cosine similarity
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_query = query_embedding / np.linalg.norm(query_embedding)

        scores = norm_embeddings @ norm_query
        
        # Get top k
        # Check if we have fewer chunks than top_k
        k = min(top_k, len(chunks))
        
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for i in top_indices:
            chunk = chunks[i].copy()
            chunk["score"] = float(scores[i])
            results.append(chunk)

        return results
    except ImportError as e:
        print(f"Embedding computation skipped: {e}")
        # Return chunks with dummy scores if we can't compute embeddings
        for chunk in chunks[:top_k]:
            chunk["score"] = 0.5  # dummy score
        return chunks[:top_k]
    except Exception as e:
        print(f"Error during retrieval: {e}")
        # Return chunks with dummy scores if we can't compute embeddings
        for chunk in chunks[:top_k]:
            chunk["score"] = 0.5  # dummy score
        return chunks[:top_k]
