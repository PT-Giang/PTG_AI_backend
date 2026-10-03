"""Run explicitly: python check_qdrant_server.py (uses configured QDRANT_URL)."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.documents import Document
from qdrant_client import QdrantClient

import rag_service
from test_document_pipeline import ConstantEmbeddings


def main():
    load_dotenv()
    collection = "task3_test_" + uuid4().hex
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"), timeout=30)
    created = False
    try:
        print("Qdrant server version:", client.info().version, flush=True)
        assert not client.collection_exists(collection)
        with patch.dict(os.environ, {"QDRANT_COLLECTION_NAME": collection}), \
             patch.object(rag_service, "get_qdrant_client", return_value=client), \
             patch.object(rag_service, "get_embeddings", return_value=ConstantEmbeddings()), \
             tempfile.TemporaryDirectory() as temp:
            rag_service.get_vector_store()
            created = True
            print("Created isolated collection:", collection, flush=True)
            for course, suffix in (("course-a", ".pdf"), ("course-b", ".docx"), ("course-a", ".pdf")):
                path = Path(temp) / (course + suffix)
                path.write_bytes(b"mock loader input")
                doc = Document(page_content="Shared topic", metadata={"source": str(path), "page": 1})
                with patch.object(rag_service, "UnstructuredFileLoader") as loader:
                    loader.return_value.load.return_value = [doc]
                    rag_service.load_and_process_document(path, course)
            assert client.count(collection, exact=True).count == 3
            for course, expected in (("course-a", 2), ("course-b", 1), ("unknown", 0)):
                results = rag_service.get_course_retriever(course).invoke("Shared topic")
                assert len(results) == expected, (course, len(results))
                assert all(doc.metadata["course_id"] == course for doc in results)
                assert all(doc.metadata["page"] == 1 for doc in results)
                print(f"PASS {course}: {len(results)} matching chunks", flush=True)
            assert len(rag_service.get_course_retriever("course-a", k=1).invoke("topic")) == 1
            assert client.count(collection, exact=True).count == 3
            print("PASS append preserves both courses; k=1 enforced", flush=True)
    finally:
        try:
            if created:
                client.delete_collection(collection)
                assert not client.collection_exists(collection)
                print("Temporary test collection removed", flush=True)
        finally:
            client.close()


if __name__ == "__main__":
    main()
