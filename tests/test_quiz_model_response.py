import unittest
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException
from src.controllers.quiz_controller import generate_quiz, QuizGenerationRequest
from src.rag.schemas import QuizSetGenerationResponse


class QuizModelResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_pydantic_response_preserves_requested_counts(self):
        result = QuizSetGenerationResponse.model_validate({
            "title": "Quiz", "sources": [], "questions": [
                {"type": "multiple_choice", "question": "Chọn A", "options": ["A", "B", "C", "D"], "correct_answer": "A", "explanation": "A đúng"},
                {"type": "short_answer", "question": "Tên giao thức?", "options": None, "correct_answer": "MQTT", "explanation": "MQTT"},
            ],
        })
        payload = QuizGenerationRequest(material_source="requirement", instruction="MQTT", question_count=2, choice_count=1, short_answer_count=1)
        with patch("src.controllers.quiz_controller.generate_one_material", AsyncMock(return_value=result)):
            response = await generate_quiz("course-1", payload)
        self.assertEqual(len(response.questions), 2)
        self.assertEqual(response.questions[0].type, "multiple_choice")
        self.assertEqual(response.questions[1].type, "short_answer")
        payload = payload.model_copy(update={"question_count": 3, "choice_count": 2})
        with patch("src.controllers.quiz_controller.generate_one_material", AsyncMock(return_value=result)):
            with self.assertRaises(HTTPException) as raised:
                await generate_quiz("course-1", payload)
        self.assertEqual(raised.exception.status_code, 502)
