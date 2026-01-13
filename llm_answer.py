import warnings
# Temporarily suppress the deprecation warning for google.generativeai
warnings.filterwarnings("ignore", message=".*google.generativeai.*deprecated.*")

# Try to import the new API first
try:
    import google.genai as genai
    GEMINI_AVAILABLE = True
    NEW_API = True
except ImportError:
    # Fall back to the old API
    try:
        import google.generativeai as genai
        GEMINI_AVAILABLE = True
        NEW_API = False
    except ImportError:
        GEMINI_AVAILABLE = False
        NEW_API = False
        print("Warning: google-generativeai module not available. LLM functionality will be limited.")

from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

def generate_structured_report(question, chunks, api_key):
    """
    Generates a structured answer using Google Gemini API or a fallback method.
    """
    if not api_key:
        return "Error: Google Gemini API Key is missing. Please enter it in the sidebar."

    if not chunks:
        return "Not enough information available to answer the question."

    # If Gemini is available, use it
    if GEMINI_AVAILABLE:
        try:
            # Configure the API key
            genai.configure(api_key=api_key)
            
            # Initialize the model
            model = genai.GenerativeModel(model_name=LLM_MODEL)
            
            context = ""
            for i, c in enumerate(chunks, 1):
                context += (
                    f"Chunk {i}\n"
                    f"Source: {c['source']}\n"
                    f"Section: {c.get('section', 'Unknown')}\n"
                    f"Relevance Score: {c.get('score', 0):.2f}\n"
                    f"Content:\n{c['text']}\n\n"
                )

            prompt = f"""You are an advanced academic research assistant.

**Goal**: Answer the research question detailed below using ONLY the provided chunks of text. 

**Instructions**:
1. Synthesize the information from the chunks into a coherent, structured report.
2. Structure your response exactly as follows:
   - **Introduction**: Briefly introduce the topic based on the context.
   - **Methodology**: Summarize methods if mentioned.
   - **Results**: Key findings.
   - **Discussion**: Analyze the findings.
   - **Conclusion**: Wrap up.
3. **Citations**: You MUST cite your sources inline using the format [Source Name, Section]. Example: [paper.pdf, Page 3].
4. **Anti-Hallucination**: If the answer is not in the chunks, state "The provided documents do not contain sufficient information to answer this part."
5. **Tone**: Academic, professional, and concise.

**Question**:
{question}

**Context Chunks**:
{context}

**Structured Answer**:
"""

            # Generate response using Gemini
            # Use the appropriate generation config based on API version
            if NEW_API:
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": LLM_TEMPERATURE,
                        "max_output_tokens": LLM_MAX_TOKENS
                    }
                )
            else:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=LLM_TEMPERATURE,
                        max_output_tokens=LLM_MAX_TOKENS
                    )
                )
            
            return response.text

        except Exception as e:
            return f"Error generating answer with Google Gemini: {str(e)}"
    else:
        # Fallback implementation when Gemini is not available
        if chunks:
            # Create a simple summary from the chunks
            context_preview = " ".join([c['text'][:200] for c in chunks[:2]])  # First 2 chunks, first 200 chars each
            return f"""
**Introduction**
Based on the provided documents, here is an analysis of: {question}

**Content Summary**
Key information extracted from the documents:
{context_preview}

**Citations**
[Sources from uploaded documents]

**Note**: Google Gemini integration is not available in this environment. Install with: `pip install google-generativeai`
            """.strip()
        else:
            return "No document content available to generate an answer."
