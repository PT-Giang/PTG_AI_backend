import unittest

from pydantic import ValidationError

from rag_service import CourseGenerationResponse, StudyQuestion


def multiple_choice(**changes):
    return dict(
        {
            "type": "multiple_choice",
            "question": "Which protocol supports publish/subscribe?",
            "hint": "Think about message brokers.",
            "core_knowledge": "MQTT uses publish/subscribe.",
            "options": ["MQTT", "HTTP", "FTP", "SMTP"],
            "correct_answer": "MQTT",
        },
        **changes,
    )


def matching(**changes):
    return dict(
        {
            "type": "matching",
            "question": "Match the terms to their definitions.",
            "hint": "Think about the role of each component.",
            "core_knowledge": "Brokers route messages to subscribers.",
            "pairs": [{"term": "Broker", "definition": "Routes messages"}],
        },
        **changes,
    )


class SchemaTests(unittest.TestCase):
    def test_course_response_round_trips_both_question_types(self):
        payload = {
            "course_id": "course-1",
            "summary": {"title": "MQTT", "overview": "Messaging protocol", "key_points": ["Publish/subscribe"]},
            "study_guide": {"title": "Hướng dẫn học tập", "steps": ["Học broker", "Thực hành topic"], "tips": ["Tự vẽ sơ đồ"]},
            "flashcards": {"topic": "MQTT", "cards": [{"front": "Broker?", "back": "Routes messages", "source_page": "lesson.pdf:1"}]},
            "study_questions": {"title": "Practice", "questions": [multiple_choice(), matching()]},
            "quiz": {"title": "Quiz", "questions": [{"question": "Protocol?", "options": ["MQTT", "HTTP", "FTP", "SMTP"], "correct_answer": "MQTT", "explanation": "Publish/subscribe"}]},
        }
        result = CourseGenerationResponse.model_validate(payload)
        restored = CourseGenerationResponse.model_validate_json(result.model_dump_json())
        self.assertEqual(restored, result)
        self.assertEqual(restored.study_questions.questions[1].pairs[0].term, "Broker")
        self.assertEqual(set(result.model_dump()), {"course_id", "summary", "study_guide", "flashcards", "study_questions", "quiz"})

    def test_multiple_choice_rejects_incomplete_or_inconsistent_answers(self):
        for changes in (
            {"options": None}, {"options": ["MQTT"]},
            {"options": ["MQTT"] * 4}, {"correct_answer": None},
            {"correct_answer": "Unknown"}, {"pairs": matching()["pairs"]},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                StudyQuestion.model_validate(multiple_choice(**changes))

    def test_matching_requires_pairs_and_rejects_choice_fields(self):
        for changes in ({"pairs": None}, {"pairs": []}, {"options": ["A"]}, {"correct_answer": "A"}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                StudyQuestion.model_validate(matching(**changes))

    def test_question_requires_supported_type_hint_and_core_knowledge(self):
        for field in ("type", "hint", "core_knowledge"):
            payload = multiple_choice()
            del payload[field]
            with self.subTest(field=field), self.assertRaises(ValidationError):
                StudyQuestion.model_validate(payload)
        with self.assertRaises(ValidationError):
            StudyQuestion.model_validate(multiple_choice(type="essay"))

    def test_json_schema_exposes_all_course_components(self):
        schema = CourseGenerationResponse.model_json_schema()
        self.assertEqual(set(schema["required"]), {"course_id", "summary", "study_guide", "flashcards", "study_questions"})
        self.assertEqual(schema["$defs"]["StudyQuestion"]["properties"]["type"]["enum"], ["multiple_choice", "matching"])


if __name__ == "__main__":
    unittest.main()
