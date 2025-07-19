import os
from langchain_community.vectorstores import FAISS

from app.components.embeddings import get_embedding_model

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)

def load_vector_store():
    try: 
        embedding_model = get_embedding_model()

        if os.path.exists(DB_FAISS_PATH):
            logger.info(f"Loading vector store from {DB_FAISS_PATH}")
            vector_store = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
            logger.info(f"Vector store loaded successfully from {DB_FAISS_PATH}")
            return vector_store
        
    except Exception as e:
        error_message = CustomException("Failed to load vector store", e)
        logger.error(str(error_message))
        

def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No text chunks to save")
        
        logger.info(f"Saving vector store to {DB_FAISS_PATH}")
        embedding_model = get_embedding_model()
        vector_store = FAISS.from_documents(text_chunks, embedding_model)
        logger.info(f"Saving vector store to {DB_FAISS_PATH}")
        vector_store.save_local(DB_FAISS_PATH)
        logger.info(f"Vector store saved successfully to {DB_FAISS_PATH}")

    except Exception as e:
        error_message = CustomException("Failed to save vector store", e)
        logger.error(str(error_message)) 
        
        