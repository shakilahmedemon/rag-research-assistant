import streamlit as st
import os
import time
from config import PAPERS_DIR
from ingest_pdfs import extract_chunks_from_pdf
from semantic_retrieval import retrieve_chunks, precompute_embeddings
from llm_answer import generate_structured_report, GEMINI_AVAILABLE
from pdf_export import create_pdf_report

def main():
    # Set page config with professional styling
    st.set_page_config(
        page_title="Advanced RAG Academic Assistant",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for advanced styling with animations
    st.markdown("""
    <style>
    /* Professional color scheme */
    :root {
        --primary-color: #1e3a8a;
        --primary-dark: #1e40af;
        --primary-light: #3b82f6;
        --secondary-color: #60a5fa;
        --accent-color: #93c5fd;
        --background-color: #f8fafc;
        --card-background: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --info-color: #0ea5e9;
    }
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        min-height: 100vh;
    }
    
    /* Header styling with gradient */
    .header-gradient {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
    }
    
    /* Animated card styling */
    .animated-card {
        background: var(--card-background);
        border-radius: 1rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .animated-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    /* Button styling with animations */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.75rem !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary-color) 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
    }
    
    .secondary-button {
        background: linear-gradient(135deg, var(--secondary-color) 0%, var(--accent-color) 100%) !important;
    }
    
    /* Input styling */
    .stTextInput > div > div > input,
    .stTextArea > div > textarea {
        border-radius: 0.75rem !important;
        border: 2px solid var(--border-color) !important;
        padding: 0.75rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > textarea:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: var(--primary-color) !important;
        border-radius: 0.5rem !important;
        height: 8px !important;
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        border: 2px dashed var(--border-color) !important;
        border-radius: 0.75rem !important;
        padding: 2rem !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader > div > div:hover {
        border-color: var(--primary-color) !important;
        background-color: #f0f9ff !important;
    }
    
    /* Custom classes */
    .section-header {
        color: var(--primary-color);
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid var(--accent-color);
        text-align: center;
    }
    
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 4px solid var(--primary-color);
        padding: 1.25rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        animation: slideInLeft 0.5s ease-out;
    }
    
    .success-box {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-left: 4px solid var(--success-color);
        padding: 1.25rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        animation: slideInRight 0.5s ease-out;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid var(--warning-color);
        padding: 1.25rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        animation: slideInLeft 0.5s ease-out;
    }
    
    .error-box {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid var(--error-color);
        padding: 1.25rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        animation: shake 0.5s ease-out;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, var(--card-background) 0%, #f1f5f9 100%);
        border-right: 1px solid var(--border-color);
    }
    
    /* Answer display styling */
    .answer-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 1rem;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    
    /* Loading spinner styling */
    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid var(--primary-color);
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 2s linear infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Metric cards styling */
    .metric-card {
        background: linear-gradient(135deg, var(--card-background) 0%, #f8fafc 100%);
        border-radius: 0.75rem;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        margin: 0.5rem;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--success-color) 0%, #34d399 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.75rem !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, var(--success-color) 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        color: var(--text-secondary);
        font-size: 0.875rem;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header section with enhanced styling
    st.markdown('<div style="text-align: center; margin-bottom: 2rem; padding: 2rem;">', unsafe_allow_html=True)
    st.markdown('<h1 class="header-gradient" style="font-size: 3rem; margin-bottom: 0.5rem;">Advanced RAG Academic Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: var(--text-secondary); font-size: 1.25rem; margin-bottom: 1rem;">Intelligent Research Paper Analysis & Question Answering System</p>', unsafe_allow_html=True)
    st.markdown('<div style="height: 4px; width: 200px; background: var(--primary-color); margin: 1rem auto; border-radius: 2px;"></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar for Configuration
    with st.sidebar:
        st.markdown('<h3 style="color: var(--primary-color); margin-bottom: 1.5rem; font-size: 1.5rem;">⚙️ Configuration</h3>', unsafe_allow_html=True)
        
        if GEMINI_AVAILABLE:
            api_key = st.text_input("🔑 Google Gemini API Key", type="password", help="Enter your Google Gemini API key from Google AI Studio")
            st.info("💡 Get your free API key from [Google AI Studio](https://aistudio.google.com/)", icon="ℹ️")
        else:
            api_key = "fallback_mode"  # Use a fallback mode when Gemini is not available
            st.warning("⚠️ Google Gemini library not installed. Running in fallback mode.", icon="⚠️")
            st.markdown("**Installation required:** `pip install google-generativeai`", unsafe_allow_html=True)

        # Add system info
        st.markdown('---')
        st.markdown('**System Information**')
        st.markdown(f'**Status:** {"✅ Active" if GEMINI_AVAILABLE else "⚠️ Limited"}')
        st.markdown(f'**LLM:** {"Google Gemini" if GEMINI_AVAILABLE else "Fallback Mode"}')
        st.markdown(f'**Version:** 1.0.0')

    # Ensure directories
    os.makedirs(PAPERS_DIR, exist_ok=True)

    # Initialize session state
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'current_step' not in st.session_state:
        st.session_state.current_step = "upload"
    if 'answer_generated' not in st.session_state:
        st.session_state.answer_generated = False
    if 'show_welcome' not in st.session_state:
        st.session_state.show_welcome = True

    # Create columns for better layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Upload Section with enhanced styling
        with st.container():
            st.markdown('<h2 class="section-header">📄 Document Processing</h2>', unsafe_allow_html=True)
            
            uploaded_files = st.file_uploader(
                "Upload Research Papers (PDF)",
                type="pdf",
                accept_multiple_files=True,
                help="Upload one or more PDF research papers for analysis"
            )

            if uploaded_files:
                # File information cards
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown(f"✅ **{len(uploaded_files)} file(s)** selected for processing")
                
                for i, file in enumerate(uploaded_files):
                    st.markdown(f"• **{file.name}** - {file.size // 1024} KB")
                st.markdown('</div>', unsafe_allow_html=True)
                
                if st.button("🔄 Process Documents", key="process_btn", help="Process uploaded PDFs and generate embeddings"):
                    # Show processing animation
                    progress_bar = st.progress(0)
                    status_container = st.empty()
                    
                    status_container.markdown('<div class="info-box">🔄 <strong>Processing documents...</strong></div>', unsafe_allow_html=True)
                    
                    total_files = len(uploaded_files)
                    
                    for i, file in enumerate(uploaded_files):
                        path = os.path.join(PAPERS_DIR, file.name)
                        with open(path, "wb") as f:
                            f.write(file.getbuffer())
                        extract_chunks_from_pdf(path)
                        progress_bar.progress((i + 1) / total_files)
                        time.sleep(0.1)  # Small delay for animation
                    
                    status_container.markdown('<div class="info-box">🧠 <strong>Computing semantic embeddings...</strong></div>', unsafe_allow_html=True)
                    time.sleep(0.5)  # Small delay for animation
                    
                    try:
                        precompute_embeddings()
                        st.session_state.processing_complete = True
                        st.session_state.current_step = "question"
                        st.session_state.show_welcome = False
                        status_container.markdown('<div class="success-box">✅ <strong>Documents processed successfully!</strong> You can now ask questions.</div>', unsafe_allow_html=True)
                    except ImportError as e:
                        st.session_state.processing_complete = True
                        st.session_state.current_step = "question"
                        st.session_state.show_welcome = False
                        status_container.markdown(f'<div class="warning-box">⚠️ <strong>Documents processed but embeddings couldn\'t be computed:</strong> {str(e)[:100]}...</div>', unsafe_allow_html=True)

    with col2:
        # Status Panel with enhanced metrics
        with st.container():
            st.markdown('<h2 class="section-header">📊 Status Panel</h2>', unsafe_allow_html=True)
            
            # Status indicators in cards
            status_col1, status_col2 = st.columns(2)
            
            with status_col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("**Documents**")
                st.markdown(f"<h3 style='color: var(--primary-color); margin: 0.5rem 0;'>{len(uploaded_files) if uploaded_files else 0}</h3>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with status_col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("**Processing**")
                status = "✅ Ready" if st.session_state.processing_complete else "⏳ Pending"
                status_color = "var(--success-color)" if st.session_state.processing_complete else "var(--warning-color)"
                st.markdown(f"<h3 style='color: {status_color}; margin: 0.5rem 0;'>{status}</h3>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Detailed status card
            st.markdown('<div class="animated-card">', unsafe_allow_html=True)
            st.markdown("**System Status**")
            
            if not uploaded_files:
                st.info("📋 No documents uploaded", icon="📋")
            elif not st.session_state.processing_complete:
                st.warning("🔄 Documents pending processing", icon="🔄")
            else:
                st.success("✅ Documents ready for analysis", icon="✅")
            
            if st.session_state.get('answer_generated', False):
                st.success("📝 Answer generated", icon="📝")
            else:
                st.info("⏳ No answer generated yet", icon="⏳")
            
            st.markdown('</div>', unsafe_allow_html=True)

    # Question and Answer Section with enhanced styling
    if st.session_state.processing_complete:
        st.markdown('<h2 class="section-header">❓ Research Question & Answer</h2>', unsafe_allow_html=True)
        
        question = st.text_area(
            "Enter your research question",
            height=150,
            placeholder="Ask a specific question about the uploaded documents... (e.g., 'What are the main findings regarding climate change in these papers?')",
            help="Enter a question related to the content of your uploaded research papers"
        )

        col_btn1, col_btn2 = st.columns([2, 1])
        
        with col_btn1:
            if st.button("🤖 Generate Answer", key="generate_btn", help="Generate answer based on uploaded documents"):
                if not api_key or api_key == "fallback_mode":
                    st.error("❌ Please enter your Google Gemini API Key in the sidebar.", icon="🔑")
                elif not question.strip():
                    st.warning("⚠️ Please enter a question.", icon="❓")
                else:
                    # Show loading animation
                    with st.spinner("🔍 Analyzing documents and generating answer..."):
                        # Simulate processing time with animation
                        progress_text = st.empty()
                        progress_text.markdown('<div class="info-box">🔍 <strong>Searching relevant information...</strong></div>', unsafe_allow_html=True)
                        time.sleep(1)
                        
                        # Retrieve
                        chunks = retrieve_chunks(question)
                        
                        if not chunks:
                            st.warning("⚠️ No relevant information found in documents. Try a different question.", icon="🔍")
                        else:
                            progress_text.markdown('<div class="info-box">🧠 <strong>Generating intelligent response...</strong></div>', unsafe_allow_html=True)
                            time.sleep(1)
                            
                            # Answer
                            answer = generate_structured_report(question, chunks, api_key)
                            
                            progress_text.markdown('<div class="info-box">📄 <strong>Creating research report...</strong></div>', unsafe_allow_html=True)
                            time.sleep(0.5)
                            
                            # Generate PDF
                            pdf_filename = "research_report.pdf"
                            create_pdf_report(answer, filename=pdf_filename)
                            
                            st.session_state.answer_generated = True
                            
                            # Clear progress text and show answer
                            progress_text.empty()
                            
                            # Display answer in a styled container
                            st.markdown('<div class="answer-container">', unsafe_allow_html=True)
                            st.markdown("<h3 style='color: var(--primary-color); margin-bottom: 1rem;'>📝 Generated Answer</h3>", unsafe_allow_html=True)
                            
                            # Add source information
                            st.markdown(f"<p style='color: var(--text-secondary); font-style: italic; margin-bottom: 1rem;'>Based on {len(chunks)} relevant document chunks</p>", unsafe_allow_html=True)
                            
                            st.markdown(answer)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Download button
                            if os.path.exists(pdf_filename):
                                st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
                                with open(pdf_filename, "rb") as f:
                                    st.download_button(
                                        label="📥 Download Research Report (PDF)",
                                        data=f,
                                        file_name=pdf_filename,
                                        mime="application/pdf",
                                        help="Download the generated research report as PDF",
                                        use_container_width=True
                                    )
                                st.markdown('</div>', unsafe_allow_html=True)
                                
        with col_btn2:
            if st.button("🔄 Reset Processing", help="Reset document processing status", type="secondary"):
                st.session_state.processing_complete = False
                st.session_state.current_step = "upload"
                st.session_state.answer_generated = False
                st.session_state.show_welcome = True
                st.rerun()

    else:
        # Enhanced welcome message when no documents are processed
        if st.session_state.show_welcome:
            st.markdown('<div class="animated-card">', unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; color: var(--primary-color); margin-bottom: 1.5rem;'>📚 Welcome to Advanced RAG Academic Assistant</h2>", unsafe_allow_html=True)
            
            # Feature cards
            col_feat1, col_feat2, col_feat3 = st.columns(3)
            
            with col_feat1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("**🔍 Semantic Search**")
                st.markdown("Find relevant information across your document collection using advanced semantic search")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_feat2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("**🤖 AI-Powered Answers**")
                st.markdown("Get intelligent, cited answers based on your research papers")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_feat3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("**📄 Professional Reports**")
                st.markdown("Generate structured academic reports in PDF format")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Workflow steps
            st.markdown("<h3 style='text-align: center; color: var(--primary-color); margin: 2rem 0 1.5rem;'>How It Works</h3>", unsafe_allow_html=True)
            
            steps_col1, steps_col2, steps_col3, steps_col4 = st.columns(4)
            
            with steps_col1:
                st.markdown('<div style="text-align: center; padding: 1rem;">', unsafe_allow_html=True)
                st.markdown("**1. 📄 Upload**")
                st.markdown("Upload your PDF research papers")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with steps_col2:
                st.markdown('<div style="text-align: center; padding: 1rem;">', unsafe_allow_html=True)
                st.markdown("**2. 🔍 Analyze**")
                st.markdown("System processes and analyzes content")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with steps_col3:
                st.markdown('<div style="text-align: center; padding: 1rem;">', unsafe_allow_html=True)
                st.markdown("**3. ❓ Ask**")
                st.markdown("Ask questions about the documents")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with steps_col4:
                st.markdown('<div style="text-align: center; padding: 1rem;">', unsafe_allow_html=True)
                st.markdown("**4. 🤖 Answer**")
                st.markdown("Get intelligent, cited answers")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("Advanced RAG Academic Assistant v1.0.0")
    st.markdown("Powered by Google Gemini AI")
    st.markdown("© 2026 Research Intelligence Systems")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
