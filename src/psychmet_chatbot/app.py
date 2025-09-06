"""Streamlit web application for PsychMet Chatbot"""

import streamlit as st
from pathlib import Path
import logging

from .chatbot import PsychMetChatbot
from .document_processor import DocumentProcessor  
from .vector_store import VectorStoreManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="PsychMet Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enable LaTeX/Math rendering
st.markdown("""
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
  },
  svg: {
    fontCache: 'global'
  }
};
</script>
""", unsafe_allow_html=True)

# Initialize session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = PsychMetChatbot()
if "messages" not in st.session_state:
    st.session_state.messages = []

def main():
    st.title("🧠 PsychMet: AI-Powered Educational Measurement Chatbot")
    st.markdown("*Personalized professional development for NCME foundational competencies*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Removed document upload for security/control reasons
        # Documents should be processed via CLI: python -m psychmet_chatbot.cli process
        
        # Clear conversation
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.chatbot.clear_memory()
            st.rerun()
        
        # Information section
        st.subheader("ℹ️ About PsychMet")
        st.info(
            "**PsychMet** is an AI-powered educational chatbot designed to support measurement professionals in developing NCME foundational competencies.\n\n"
            "**Key Features:**\n"
            "• Personalized professional development guidance\n"
            "• NCME foundational competencies framework integration\n"
            "• Item Response Theory (IRT) expertise\n"
            "• Retrieval Augmented Generation (RAG) technology\n"
            "• Tailored learning experiences for diverse backgrounds\n\n"
            "**Target Competency Domains:**\n"
            "• Communication and Collaboration\n"
            "• Technical, Statistical, and Computational\n"
            "• Educational Measurement Competencies\n\n"
            "*Built with GPT-4 and FAISS vector search for accurate, contextually relevant responses.*"
        )
    
    # Main chat interface
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about NCME competencies, IRT, psychometrics, or professional development..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.chat(prompt)
                answer = response["answer"]
                
                st.markdown(answer)
                
                # Show sources if available
                if response["source_documents"]:
                    with st.expander("📖 Sources"):
                        for i, doc in enumerate(response["source_documents"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.text(doc.page_content[:500] + "...")
                            if hasattr(doc, 'metadata'):
                                st.caption(f"Page: {doc.metadata.get('page', 'N/A')}")
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()