import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import *

logger = get_logger(__name__)

def load_pdf_files():
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException("Data path does not exist")

        logger.info(f"Loading PDF files from {DATA_PATH}")

        loader = PyPDFDirectoryLoader(DATA_PATH, glob="*.pdf")

        documents = loader.load()

        if not documents:
            logger.warning("No PDF files found in the data path")
        else:
            logger.info(f"Loaded {len(documents)} PDF files")

        return documents

    except Exception as e:
        error_message = CustomException("Failed to load PDF files")
        logger.error(str(error_message))
        return []


def create_text_chunks(documents):
    try:
        if not documents:
            raise CustomException("No documents to create text chunks")

        logger.info(f"Creating text chunks from documents, total documents: {len(documents)}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        text_chunks = text_splitter.split_documents(documents)

        logger.info(f"Created {len(text_chunks)} text chunks")

        return text_chunks

    except Exception as e:
        error_message = CustomException("Failed to create text chunks", e)
        logger.error(str(error_message))
        return []
