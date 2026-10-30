import os
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import List
from uuid import UUID, uuid5

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

from src.rag.chunker import chunk_documents
from src.rag.config import qdrant_collection_name, qdrant_url
from src.rag.embeddings import get_embeddings
from src.rag.loader import load_document


_collection_lock = Lock()


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=qdrant_url(), timeout=30)


def get_vector_store() -> QdrantVectorStore:
    collection_name = qdrant_collection_name()
    client = get_qdrant_client()
    embeddings = get_embeddings()
    with _collection_lock:
        if not client.collection_exists(collection_name):
            vector_size = len(embeddings.embed_query("dimension probe"))
            try:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
            except Exception:
                if not client.collection_exists(collection_name):
                    raise
    return QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)


def validate_course_id(course_id: str) -> None:
    if not isinstance(course_id, str) or not course_id.strip():
        raise ValueError("course_id must be a non-empty string")


def load_and_process_document(file_path: str | Path, course_id: str, source_name: str | None = None, document_id: UUID | None = None) -> List[Document]:
    validate_course_id(course_id)
    documents = load_document(file_path, source_name)
    chunks = chunk_documents(documents)
    if not chunks:
        raise ValueError("Document produced no text chunks")
    for chunk in chunks:
        chunk.metadata["course_id"] = course_id
        if document_id is not None:
            chunk.metadata["document_id"] = str(document_id)
    get_vector_store().add_documents(chunks)
    return chunks


def get_course_retriever(course_id: str, k: int = 6, document_id: UUID | None = None):
    validate_course_id(course_id)
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    conditions = [FieldCondition(key="metadata.course_id", match=MatchValue(value=course_id))]
    if document_id is not None:
        conditions.append(FieldCondition(key="metadata.document_id", match=MatchValue(value=str(document_id))))
    return get_vector_store().as_retriever(
        search_kwargs={"k": k, "filter": Filter(must=conditions)}
    )


def index_text_document(course_id: str, document_id: UUID, source_name: str, text: str) -> int:
    validate_course_id(course_id)
    if not source_name.strip() or not text.strip():
        raise ValueError("source_name and text must be non-empty")
    document = Document(
        page_content=text.strip(),
        metadata={"source_name": Path(source_name.replace("\\", "/")).name, "document_id": str(document_id)},
    )
    chunks = chunk_documents([document])
    if not chunks:
        raise ValueError("Document produced no text chunks")
    ids = []
    for index, chunk in enumerate(chunks):
        chunk.metadata["course_id"] = course_id
        chunk.metadata["document_id"] = str(document_id)
        ids.append(str(uuid5(document_id, f"{index}:{chunk.page_content}")))
    get_vector_store().add_documents(chunks, ids=ids)
    return len(chunks)
