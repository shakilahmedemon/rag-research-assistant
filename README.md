## Advanced RAG Academic Assistant

An intelligent, professional-grade Research Paper Analysis & Question Answering system. This tool utilizes a Retrieval-Augmented Generation (RAG) pipeline to help users synthesize information from complex academic documents.


 ## Key Features
 
1. Multi-PDF Processing: Upload and index multiple research papers simultaneously.
2. Semantic Retrieval: Uses all-MiniLM-L6-v2 embeddings to find the most relevant context for your questions, not just keyword matches.
3. Gemini-Powered Synthesis: Leverages Google Gemini (Pro) to generate structured, academic-style reports (Introduction, Methodology, Results, Discussion, Conclusion).
4. Automatic Citations: Every generated answer includes inline citations (e.g., [paper.pdf, Chunk 1]) to prevent hallucination.
5. PDF Export: Download your generated research report as a professionally formatted PDF.
6. Modern UI: Built with a custom-styled Streamlit interface featuring CSS animations and responsive layouts.


## Technical Architecture

1. Ingestion: pypdf extracts text from uploaded PDFs.
2. Chunking: Documents are split into 1000-character segments with a 200-character overlap to preserve context.
3. Embeddings: sentence-transformers creates vector representations of text chunks stored in numpy arrays.
4. Retrieval: Performs Cosine Similarity search to find the top $k$ relevant chunks.
5. Generation: Chunks are passed to the gemini-pro model with strict "Anti-Hallucination" prompting.


## Getting Started
### 1. Prerequisites

1. Python 3.9 or higher.
2. A Google AI Studio API Key.


### 2. Installation
Clone the repository and install the dependencies:

- git clone https://github.com/shakilahmedemon/rag-research-assistant
- cd advanced-rag-academic-assistant
- pip install -r requirements.txt


### 3. Configuration
The system uses a config.py file for global settings. You can adjust:

1. CHUNK_SIZE: Default 1000.
2. TOP_K: Number of document chunks to retrieve (Default 5).
3. LLM_MODEL: Default "gemini-pro".


### 4. Running the App

- streamlit run app.py


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Author

AHMED MD SHAKIL

Studying Master's in Software Engineering at Yangzhou University, China
