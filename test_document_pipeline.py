import os
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.qdrant import QdrantVectorStoreError
from qdrant_client import QdrantClient, models

import rag_service


class ConstantEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


class DocumentPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.client = QdrantClient(location=":memory:")
        self.addCleanup(self.client.close)
        self.client.create_collection(
            "courses", vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE)
        )
        self.store = QdrantVectorStore(self.client, "courses", embedding=ConstantEmbeddings())
        for name, value in (("get_vector_store", self.store), ("get_embeddings", ConstantEmbeddings())):
            patcher = patch.object(rag_service, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def load(self, course_id, suffix=".pdf"):
        path = Path(self.temp.name) / (course_id + suffix)
        path.write_bytes(b"mock document")
        doc = Document(page_content="Shared topic", metadata={"source": str(path), "page": 2})
        with patch.object(rag_service, "UnstructuredFileLoader") as loader:
            loader.return_value.load.return_value = [doc]
            chunks = rag_service.load_and_process_document(path, course_id)
            loader.assert_called_once_with(str(path), strategy="fast", languages=["vie", "eng"])
        return chunks

    def test_pdf_and_docx_preserve_metadata_and_isolate_courses(self):
        first = self.load("course-a", ".pdf")
        self.load("course-b", ".docx")
        self.assertEqual(first[0].metadata["course_id"], "course-a")
        self.assertEqual(first[0].metadata["page"], 2)
        for course in ("course-a", "course-b"):
            results = rag_service.get_course_retriever(course).invoke("Shared topic")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].metadata["course_id"], course)
        self.assertEqual(rag_service.get_course_retriever("unknown").invoke("Shared topic"), [])

    def test_append_keeps_previous_courses_and_obeys_k(self):
        self.load("course-a")
        self.load("course-a")
        self.load("course-b")
        self.assertEqual(self.client.count("courses").count, 3)
        self.assertEqual(len(rag_service.get_course_retriever("course-a", k=1).invoke("topic")), 1)

    def test_invalid_input_is_rejected_before_loading(self):
        with self.assertRaises(ValueError):
            rag_service.load_and_process_document("file.txt", "course-a")
        with self.assertRaises(FileNotFoundError):
            rag_service.load_and_process_document(Path(self.temp.name) / "missing.pdf", "course-a")
        for course in ("", "   ", None, 123):
            with self.subTest(course=course), self.assertRaises(ValueError):
                rag_service.get_course_retriever(course)
        for k in (0, -1, True, 1.5):
            with self.subTest(k=k), self.assertRaises(ValueError):
                rag_service.get_course_retriever("course-a", k=k)

    def test_empty_document_does_not_write_vectors(self):
        path = Path(self.temp.name) / "empty.pdf"
        path.write_bytes(b"mock")
        with patch.object(rag_service, "UnstructuredFileLoader") as loader:
            loader.return_value.load.return_value = [Document(page_content="  ")]
            with self.assertRaises(ValueError):
                rag_service.load_and_process_document(path, "course-a")
        self.assertEqual(self.client.count("courses").count, 0)


class VectorStoreTests(unittest.TestCase):
    def test_creates_collection_then_reuses_existing_data(self):
        with closing(QdrantClient(location=":memory:")) as client:
            with patch.object(rag_service, "get_qdrant_client", return_value=client), \
                 patch.object(rag_service, "get_embeddings", return_value=ConstantEmbeddings()), \
                 patch.object(rag_service, "load_dotenv"), \
                 patch.dict(os.environ, {"QDRANT_COLLECTION_NAME": "new-courses"}):
                store = rag_service.get_vector_store()
                store.add_documents([Document(page_content="Keep me", metadata={"course_id": "old"})])
                reopened = rag_service.get_vector_store()
                self.assertEqual(client.count("new-courses").count, 1)
                self.assertEqual(reopened.similarity_search("topic")[0].page_content, "Keep me")

    def test_incompatible_existing_collection_is_not_recreated(self):
        with closing(QdrantClient(location=":memory:")) as client:
            client.create_collection("incompatible", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
            with patch.object(rag_service, "get_qdrant_client", return_value=client), \
                 patch.object(rag_service, "get_embeddings", return_value=ConstantEmbeddings()), \
                 patch.object(rag_service, "load_dotenv"), \
                 patch.dict(os.environ, {"QDRANT_COLLECTION_NAME": "incompatible"}):
                with self.assertRaises(QdrantVectorStoreError):
                    rag_service.get_vector_store()
            self.assertEqual(client.get_collection("incompatible").config.params.vectors.size, 2)


if __name__ == "__main__":
    unittest.main()
