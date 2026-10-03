import asyncio
import json
import threading
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

import rag_service


RESULTS = {
    "summary": {"title": "MQTT", "overview": "Messaging", "key_points": ["Publish/subscribe"]},
    "flashcards": {"topic": "MQTT", "cards": [{"front": "Broker?", "back": "Routes messages", "source_page": "lesson.pdf:1"}]},
    "study_questions": {"title": "Practice", "questions": [{
        "type": "matching", "question": "Match", "hint": "Roles", "core_knowledge": "Messaging",
        "pairs": [{"term": "Broker", "definition": "Routes messages"}],
        "options": None, "correct_answer": None,
    }]},
    "quiz": {"title": "Quiz", "questions": [{"question": "Protocol?", "options": ["MQTT", "HTTP", "FTP", "SMTP"], "correct_answer": "MQTT", "explanation": "Publish/subscribe"}]},
}


class ChainTests(unittest.IsolatedAsyncioTestCase):
    async def test_topic_chain_uses_general_knowledge_without_document_instructions(self):
        prompts = []

        def answer(prompt):
            prompts.append(prompt.to_string())
            return json.dumps(RESULTS["summary"])

        with patch.object(rag_service, "ChatOpenAI", return_value=RunnableLambda(answer)):
            chain = rag_service.get_teacher_chains()["summary"]
        await chain.ainvoke("Tôi muốn học HTML5")
        self.assertIn("HTML5", prompts[0])
        self.assertIn("kiến thức", prompts[0])
        self.assertNotIn("Chỉ dùng kiến thức trong ngữ cảnh", prompts[0])

    async def test_all_chains_use_context_requirement_and_parse_json(self):
        prompts = []

        def answer(prompt):
            text = prompt.to_string()
            prompts.append(text)
            for marker, key in (("CourseSummary", "summary"), ("FlashcardSet", "flashcards"),
                                ("StudyQuestionSet", "study_questions"), ("QuizSet", "quiz")):
                if marker in text:
                    return json.dumps(RESULTS[key])
            raise AssertionError("Prompt does not identify its schema")

        retriever = RunnableLambda(lambda _: [Document(page_content="MQTT context", metadata={"source": "lesson.pdf", "page": 1})])
        with patch.object(rag_service, "ChatOpenAI", return_value=RunnableLambda(answer)) as llm:
            chains = rag_service.get_teacher_chains(retriever)
        self.assertEqual(set(chains), set(RESULTS))
        self.assertEqual(llm.call_args.kwargs["model"], "deepseek-v4-flash")
        self.assertEqual(llm.call_args.kwargs["temperature"], 0.2)
        for key, chain in chains.items():
            self.assertEqual(await chain.ainvoke("Explain MQTT"), RESULTS[key])
        self.assertTrue(all("MQTT context" in text and "Explain MQTT" in text for text in prompts))

    async def test_valid_json_with_invalid_schema_is_rejected(self):
        retriever = RunnableLambda(lambda _: [Document(page_content="MQTT")])
        with patch.object(rag_service, "ChatOpenAI", return_value=RunnableLambda(lambda _: '{"title": "Incomplete"}')):
            chain = rag_service.get_teacher_chains(retriever)["summary"]
        with self.assertRaises(ValidationError):
            await chain.ainvoke("MQTT")

    async def test_non_json_output_is_rejected(self):
        retriever = RunnableLambda(lambda _: [Document(page_content="MQTT")])
        with patch.object(rag_service, "ChatOpenAI", return_value=RunnableLambda(lambda _: "not JSON")):
            chain = rag_service.get_teacher_chains(retriever)["quiz"]
        with self.assertRaises(OutputParserException):
            await chain.ainvoke("MQTT")


class GenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_topic_only_skips_document_embedding_and_qdrant(self):
        chains = {key: RunnableLambda(lambda _, value=value: value) for key, value in RESULTS.items()}
        with patch.object(rag_service, "load_and_process_document") as index, \
             patch.object(rag_service, "get_course_retriever") as retriever, \
             patch.object(rag_service, "get_teacher_chains", return_value=chains) as build:
            result = await rag_service.generate_course_materials(None, "html5", "Tôi muốn học HTML5")
        index.assert_not_called()
        retriever.assert_not_called()
        build.assert_called_once_with(None)
        self.assertEqual(result.course_id, "html5")

    async def test_four_chains_run_concurrently_after_indexing_off_event_loop(self):
        started = set()
        ready = asyncio.Event()
        main_thread = threading.get_ident()
        indexed = threading.Event()

        def index(*args):
            self.assertNotEqual(threading.get_ident(), main_thread)
            self.assertEqual(args, ("lesson.pdf", "course-1"))
            indexed.set()

        def make_chain(key):
            async def run(requirement):
                self.assertTrue(indexed.is_set())
                self.assertEqual(requirement, "Explain MQTT")
                started.add(key)
                if len(started) == 4:
                    ready.set()
                await asyncio.wait_for(ready.wait(), timeout=5)
                return RESULTS[key]
            return RunnableLambda(run)

        with patch.object(rag_service, "load_and_process_document", side_effect=index), \
             patch.object(rag_service, "get_course_retriever") as retriever, \
             patch.object(rag_service, "get_teacher_chains", return_value={key: make_chain(key) for key in RESULTS}):
            response = await rag_service.generate_course_materials("lesson.pdf", "course-1", "Explain MQTT")
        retriever.assert_called_once_with("course-1")
        self.assertEqual(response.model_dump(), {"course_id": "course-1", **RESULTS})

    async def test_failure_cancels_other_chains(self):
        ready = asyncio.Event()
        started = set()
        cancelled = set()

        def make_chain(key):
            async def run(_):
                started.add(key)
                if len(started) == 4:
                    ready.set()
                await ready.wait()
                if key == "quiz":
                    raise RuntimeError("LLM failed")
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.add(key)
            return RunnableLambda(run)

        with patch.object(rag_service, "load_and_process_document"), \
             patch.object(rag_service, "get_course_retriever"), \
             patch.object(rag_service, "get_teacher_chains", return_value={key: make_chain(key) for key in RESULTS}):
            with self.assertRaisesRegex(RuntimeError, "LLM failed"):
                await asyncio.wait_for(rag_service.generate_course_materials("lesson.pdf", "course-1", "MQTT"), 5)
        self.assertEqual(cancelled, set(RESULTS) - {"quiz"})

    async def test_invalid_requirement_does_not_index(self):
        with patch.object(rag_service, "load_and_process_document") as index:
            for requirement in ("", "  ", None):
                with self.subTest(requirement=requirement), self.assertRaises(ValueError):
                    await rag_service.generate_course_materials("lesson.pdf", "course-1", requirement)
            index.assert_not_called()

    async def test_indexing_failure_does_not_start_generation(self):
        with patch.object(rag_service, "load_and_process_document", side_effect=ValueError("Empty document")), \
             patch.object(rag_service, "get_teacher_chains") as chains:
            with self.assertRaisesRegex(ValueError, "Empty document"):
                await rag_service.generate_course_materials("lesson.pdf", "course-1", "MQTT")
            chains.assert_not_called()


if __name__ == "__main__":
    unittest.main()
