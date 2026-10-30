from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from src.rag.embeddings import get_embeddings


def chunk_documents(documents: list[Document]) -> list[Document]:
    chunker = SemanticChunker(
        embeddings=get_embeddings(), breakpoint_threshold_amount=0.85,
    )
    return [chunk for chunk in chunker.split_documents(documents) if chunk.page_content.strip()]
