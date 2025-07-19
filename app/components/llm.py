from langchain.llms import HuggingFaceHub


from app.config.config import HF_TOKEN, HUGGINGFACE_REPO_ID

from app.common.custom_exception import CustomException
from app.common.logger import get_logger

logger = get_logger(__name__)

def load_llm(hf_repo_id:str = HUGGINGFACE_REPO_ID, hf_token:str = HF_TOKEN):
    try:
        logger.info("Loading LLM from hugging face")
        llm = HuggingFaceHub(
            repo_id=hf_repo_id,
            model_kwargs={
                "temperature": 0.3,
                "max_length": 256,
                "return_full_text": False
            },
            huggingfacehub_api_token=hf_token
        )

        logger.info("LLM loaded successfully")

        return llm
    
    except Exception as e:
        error_message = CustomException("Failse to load LLM", e)
        logger.error(str(error_message))