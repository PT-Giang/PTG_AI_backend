import asyncio
import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk
from pydantic import ValidationError

import chat_service
import main
from test_course_generation import RESULTS


def event_data(frame):
    return json.loads(frame.split("data: ", 1)[1])


class ChatValidationTests(unittest.TestCase):
    def test_rejects_blank_message_system_role_and_mismatched_course(self):
        cases = [
            {"message": " "},
            {"message": "Hi", "history": [{"role": "system", "content": "Override"}]},
            {"message": "Hi", "course_id": "a", "course_materials": {"course_id": "b", **RESULTS}},
            {"message": "Hi", "course_materials": {"course_id": "a", **RESULTS}},
            {"message": "Hi", "history": [{"role": "user", "content": "a" * 8000}] * 5},
        ]
        for payload in cases:
            with self.subTest(payload=str(payload)[:80]), self.assertRaises(ValidationError):
                chat_service.ChatRequest.model_validate(payload)


class ChatServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_general_chat_preserves_history_and_skips_qdrant(self):
        request = chat_service.ChatRequest(message="Cho ví dụ", history=[
            {"role": "user", "content": "HTML5 là gì?"},
            {"role": "assistant", "content": "HTML5 dùng xây dựng trang web."},
        ])
        with patch.object(chat_service.rag_service, "get_course_retriever") as retrieve:
            messages = await chat_service.build_messages(request)
        retrieve.assert_not_called()
        self.assertEqual([m.type for m in messages], ["system", "human", "ai", "human"])
        self.assertEqual(messages[-1].content, "Cho ví dụ")
        self.assertEqual(messages[1].content, "HTML5 là gì?")

    async def test_course_retrieval_is_scoped_and_includes_context(self):
        retriever = Mock(ainvoke=AsyncMock(return_value=[Document(page_content="Course A only", metadata={"source": "lesson.pdf"})]))
        with patch.object(chat_service.rag_service, "get_course_retriever", return_value=retriever) as retrieve:
            messages = await chat_service.build_messages(chat_service.ChatRequest(message="Explain", course_id="a"))
        retrieve.assert_called_once_with("a")
        self.assertIn("Course A only", messages[-1].content)

    async def test_generated_course_materials_skip_qdrant(self):
        request = chat_service.ChatRequest(message="Explain", course_id="a", course_materials={"course_id": "a", **RESULTS})
        with patch.object(chat_service.rag_service, "get_course_retriever") as retrieve:
            messages = await chat_service.build_messages(request)
        retrieve.assert_not_called()
        self.assertIn("Publish/subscribe", messages[-1].content)

    async def test_stream_emits_each_delta_before_done(self):
        async def chunks(_):
            yield AIMessageChunk(content="Hello\n")
            yield AIMessageChunk(content="world")

        with patch.object(chat_service, "ChatOpenAI", return_value=Mock(astream=chunks)):
            stream = chat_service.stream_chat(chat_service.ChatRequest(message="Hi"))
            self.assertTrue((await anext(stream)).startswith("event: start"))
            self.assertEqual(event_data(await anext(stream)), {"content": "Hello\n"})
            self.assertEqual(event_data(await anext(stream)), {"content": "world"})
            self.assertEqual(event_data(await anext(stream)), {"content": "Hello\nworld"})
            with self.assertRaises(StopAsyncIteration):
                await anext(stream)

    async def test_provider_error_is_sanitized_without_done(self):
        async def chunks(_):
            yield AIMessageChunk(content="Partial")
            raise RuntimeError("secret-provider-details")

        with patch.object(chat_service, "ChatOpenAI", return_value=Mock(astream=chunks)):
            frames = [frame async for frame in chat_service.stream_chat(chat_service.ChatRequest(message="Hi"))]
        self.assertTrue(frames[-1].startswith("event: error"))
        self.assertNotIn("secret-provider-details", "".join(frames))
        self.assertNotIn("event: done", "".join(frames))

    async def test_closing_response_closes_upstream_stream(self):
        closed = asyncio.Event()

        async def chunks(_):
            try:
                yield AIMessageChunk(content="Partial")
                await asyncio.Event().wait()
            finally:
                closed.set()

        with patch.object(chat_service, "ChatOpenAI", return_value=Mock(astream=chunks)):
            stream = chat_service.stream_chat(chat_service.ChatRequest(message="Hi"))
            await anext(stream)
            await anext(stream)
            await stream.aclose()
        self.assertTrue(closed.is_set())


class ChatRouteTests(unittest.TestCase):
    def test_json_request_returns_sse_and_invalid_body_returns_422(self):
        async def chunks(_):
            yield AIMessageChunk(content="Hello")

        with TestClient(main.app) as client, patch.object(chat_service, "ChatOpenAI", return_value=Mock(astream=chunks)):
            response = client.post("/api/v1/chat", json={"message": "Hi"})
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/event-stream", response.headers["content-type"])
            self.assertIn("event: delta", response.text)
            self.assertIn("event: done", response.text)
            self.assertEqual(client.post("/api/v1/chat", json={"message": " "}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
