import os

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from app.components.vectore_store import load_vector_store
from app.components.llm import load_llm

from app.common.custom_exception import CustomException
from app.common.logger import get_logger

logger = get_logger(__name__)


CUSTOM_PROMPT_TEMPLATE = """ Answer the following medical question in 2-3 lines maximum using only the information provided in the context

Context:
{context}

Question:
{question}

Answer:
"""


def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])


def create_qa_chain():
    try:
        logger.info("Loading vector store ...")
        db = load_vector_store()

        if db is None:
            raise CustomException("Vector store not present or empty.")

        llm = load_llm()

        if llm is None:
            raise CustomException("LLM not loaded")

        qa_chain = RetrievalQA.from_chain_type(
            llm = llm,
            chain_type = "stuff",
            retriever = db.as_retriever(search_kwargs={'k':1}),
            return_source_documents = False,
            chain_type_kwargs = {'prompt': set_custom_prompt()}
        )

        logger.info("Successfully created the QA chain")
        return qa_chain

    except Exception as e:
        error_message = CustomException("Failed to make a QA chain", e)
        logger.error(str(error_message))