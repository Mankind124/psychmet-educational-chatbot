"""Main chatbot implementation using LangChain"""

import logging
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from .config import config
from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class PsychMetChatbot:
    """Main chatbot class for psychometric education"""
    
    def __init__(self, vector_store=None):
        config.validate()
        if vector_store is None:
            self.vector_manager = VectorStoreManager()
            vector_store = self.vector_manager.create_or_load_store()
        self.vector_store = vector_store
        
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            openai_api_key=config.OPENAI_API_KEY
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.qa_chain = None
        self._setup_qa_chain()
    
    def _setup_qa_chain(self):
        """Setup the QA chain with custom prompt"""
        prompt_template = """You are an expert psychometrics educator assistant. 
        Use the following context to answer questions about psychological measurement, 
        test construction, reliability, validity, and statistical methods in psychology.
        
        IMPORTANT FORMATTING RULES:
        - Use proper LaTeX notation for mathematical equations (wrap in $ for inline or $$ for display)
        - For example: 1PL model as $P(X_{{ij}} = 1) = \\frac{{e^{{\\theta_i - b_j}}}}{{1 + e^{{\\theta_i - b_j}}}}$
        - For Cronbach's alpha: $\\alpha = \\frac{{k}}{{k-1}}\\left(1 - \\frac{{\\sum s_i^2}}{{s_t^2}}\\right)$
        - Use proper statistical notation and Greek letters
        
        If you don't know the answer based on the context, say so honestly.
        Always explain concepts clearly with examples when appropriate.
        
        Context: {context}
        
        Question: {question}
        
        Educational Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def chat(self, question: str) -> Dict[str, Any]:
        """Process a user question and return answer with sources"""
        try:
            response = self.qa_chain({"question": question})
            return {
                "answer": response["answer"],
                "source_documents": response.get("source_documents", []),
                "chat_history": self.memory.chat_memory.messages
            }
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return {
                "answer": f"I encountered an error: {str(e)}",
                "source_documents": [],
                "chat_history": []
            }
    
    def get_response_with_context(self, question: str) -> Dict[str, Any]:
        """Get response with context documents for evaluation purposes"""
        try:
            response = self.qa_chain({"question": question})
            return {
                "answer": response["answer"],
                "source_documents": response.get("source_documents", []),
                "chat_history": self.memory.chat_memory.messages
            }
        except Exception as e:
            logger.error(f"Error in get_response_with_context: {e}")
            return {
                "answer": f"I encountered an error: {str(e)}",
                "source_documents": [],
                "chat_history": []
            }
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        logger.info("Conversation memory cleared")