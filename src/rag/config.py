import os

from dotenv import load_dotenv


def load_config() -> None:
    load_dotenv()


def qdrant_url() -> str:
    load_config()
    return os.getenv("QDRANT_URL", "http://localhost:6333")


def qdrant_collection_name() -> str:
    load_config()
    return os.getenv("QDRANT_COLLECTION_NAME", "papers_collection")


def llm_settings() -> dict:
    load_config()
    return {
        "model": "deepseek-v4-flash",
        "api_key": os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        "temperature": 0.2,
    }
