import json
import os
from pypdf import PdfReader
from config import CHUNKS_FILE, CHUNK_SIZE, CHUNK_OVERLAP, MEMORY_DIR

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Splits text into overlapping chunks."""
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Break if we've reached the end
        if end >= len(text):
            break
            
        start += (chunk_size - overlap)
        
    return chunks

def extract_chunks_from_pdf(pdf_path: str) -> int:
    """
    Extract text from a PDF, chunk it, and store in chunks.json.
    Returns number of new chunks extracted.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return 0

    full_text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n\n"
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return 0
    
    # Crude sanitation
    # Replace multiple spaces/newlines could go here if needed, 
    # but simplest is often just to carry on.
    
    text_chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
    
    new_chunks = []
    base_name = os.path.basename(pdf_path)
    
    for i, txt in enumerate(text_chunks):
        new_chunks.append({
            "chunk_id": f"{base_name}_chunk_{i}",
            "source": base_name,
            "section": f"Chunk {i+1}", # Approximate location
            "type": "text",
            "text": txt
        })

    # Load existing
    existing_chunks = []
    if os.path.exists(CHUNKS_FILE):
        try:
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                existing_chunks = json.load(f)
        except json.JSONDecodeError:
            print("Warning: chunks.json was corrupted. Overwriting.")
            existing_chunks = []

    # Update or Append? 
    # Current logic simply appends. In a real app we might want to deduplicate by source.
    # For now, let's remove old chunks from the same source before adding new ones
    # to avoid duplicates if re-uploaded.
    filtered_chunks = [c for c in existing_chunks if c["source"] != base_name]
    
    final_chunks = filtered_chunks + new_chunks

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, indent=2)

    return len(new_chunks)
