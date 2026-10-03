"""Run real loader, embedding and Qdrant: python check_real_documents.py."""
import os
import time
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

import rag_service


def main():
    load_dotenv()
    files = sorted(path for path in Path("papers").iterdir() if path.suffix.lower() in {".docx", ".pdf"})
    if not files:
        raise RuntimeError("No PDF/DOCX files found in papers/")
    collection = "real_docs_test_" + uuid4().hex
    previous = os.environ.get("QDRANT_COLLECTION_NAME")
    os.environ["QDRANT_COLLECTION_NAME"] = collection
    client = rag_service.get_qdrant_client()
    created = False
    started = time.monotonic()
    try:
        print("Qdrant version:", client.info().version, flush=True)
        assert not client.collection_exists(collection)
        print("Loading multilingual-e5-small and creating test collection...", flush=True)
        rag_service.get_vector_store()
        created = True
        print("Test collection:", collection, flush=True)
        total = 0
        for index, path in enumerate(files):
            course_id = f"real-course-{index}"
            print("Processing:", path.name, flush=True)
            chunks = rag_service.load_and_process_document(path, course_id)
            total += len(chunks)
            assert chunks and all(chunk.metadata["course_id"] == course_id for chunk in chunks)
            assert client.count(collection, exact=True).count == total
            results = rag_service.get_course_retriever(course_id, k=3).invoke(chunks[0].page_content[:200])
            assert len(results) == min(3, len(chunks))
            assert all(doc.metadata["course_id"] == course_id for doc in results)
            assert all(Path(doc.metadata["source"]).name == path.name for doc in results)
            print(f"PASS {path.name}: {len(chunks)} chunks, retrieved {len(results)}, source/course verified", flush=True)
        for index in range(len(files)):
            results = rag_service.get_course_retriever(f"real-course-{index}").invoke("Zero Trust vocabulary")
            assert results and all(doc.metadata["course_id"] == f"real-course-{index}" for doc in results)
        assert rag_service.get_course_retriever("unknown-course").invoke("Zero Trust") == []
        print(f"PASS all {len(files)} documents: {total} points; isolation verified; {time.monotonic() - started:.1f}s", flush=True)
    finally:
        try:
            if created:
                client.delete_collection(collection)
                assert not client.collection_exists(collection)
                print("Temporary collection removed", flush=True)
        finally:
            client.close()
            rag_service.get_qdrant_client.cache_clear()
            if previous is None:
                os.environ.pop("QDRANT_COLLECTION_NAME", None)
            else:
                os.environ["QDRANT_COLLECTION_NAME"] = previous


if __name__ == "__main__":
    main()
