import unittest
from unittest.mock import patch
from uuid import uuid4
from fastapi import HTTPException

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

import src.controllers.common as generation_common
from src.controllers.common import ChainGenerationRequest
from src.rag.schemas import build_source_metadata


class GenerationSourceTests(unittest.TestCase):
    def test_source_metadata_uses_retrieved_document_title_page_and_excerpt(self):
        document = Document(
            page_content="MQTT dùng broker để định tuyến thông điệp.",
            metadata={"source": "C:/temp/random-upload.pdf", "source_name": "lesson.pdf", "page": 2},
        )

        source = build_source_metadata([document])[0]

        self.assertEqual(source.source_type, "document")
        self.assertEqual(source.title, "lesson.pdf")
        self.assertEqual(source.excerpt, "Trang 2: MQTT dùng broker để định tuyến thông điệp.")
        self.assertIsNotNone(source.retrieved_at.tzinfo)
        self.assertFalse(source.is_mock)
        self.assertIsNone(source.document_id)


class GenerationControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_document_context_never_calls_llm(self):
        request = ChainGenerationRequest(material_source="document", instruction="Tạo quiz", document_id=uuid4())
        with patch.object(generation_common.rag_service, "get_course_retriever", return_value=RunnableLambda(lambda _: [])) as retrieve, \
             patch.object(generation_common.rag_service, "get_teacher_chains") as chains:
            with self.assertRaises(HTTPException) as error:
                await generation_common.generate_one_material("course-1", "quiz", request)
        self.assertEqual(error.exception.status_code, 422)
        retrieve.assert_called_once_with("course-1", 6, request.document_id)
        chains.assert_not_called()

    async def test_document_generation_returns_sources_from_the_same_retrieval_context(self):
        documents = [Document(
            page_content="MQTT dùng broker để định tuyến thông điệp.",
            metadata={"source_name": "lesson.pdf", "page": 2},
        )]
        retrieval = RunnableLambda(lambda _: documents)
        expected_output = {"title": "MQTT", "overview": "Messaging", "key_points": ["Broker"]}
        request = ChainGenerationRequest(material_source="document", instruction="Giải thích MQTT")

        def build_chains(context_retriever, chain_names):
            self.assertEqual(context_retriever.invoke(request.instruction), documents)
            self.assertEqual(chain_names, ["summary"])
            return {"summary": RunnableLambda(lambda _: expected_output)}

        with patch.object(generation_common.rag_service, "get_course_retriever", return_value=retrieval), \
             patch.object(generation_common.rag_service, "get_teacher_chains", side_effect=build_chains):
            result = await generation_common.generate_one_material("course-1", "summary", request)

        self.assertEqual(result.title, "MQTT")
        self.assertEqual(result.sources[0].title, "lesson.pdf")
        self.assertEqual(result.sources[0].excerpt, "Trang 2: MQTT dùng broker để định tuyến thông điệp.")

    async def test_requirement_generation_returns_no_fabricated_sources(self):
        request = ChainGenerationRequest(material_source="requirement", instruction="Học MQTT")
        with patch.object(generation_common.rag_service, "get_course_retriever") as get_retriever, \
             patch.object(generation_common.rag_service, "get_teacher_chains", return_value={
                 "summary": RunnableLambda(lambda _: {"title": "MQTT", "overview": "Messaging", "key_points": ["Broker"]})
             }):
            result = await generation_common.generate_one_material("course-1", "summary", request)
        get_retriever.assert_not_called()
        self.assertEqual(result.sources, [])


if __name__ == "__main__":
    unittest.main()
