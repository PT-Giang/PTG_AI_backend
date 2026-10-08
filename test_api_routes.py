import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

import main
from rag_service import CourseGenerationResponse
from test_course_generation import RESULTS


class APIRouteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.uploads = Path(self.temp.name)
        patcher = patch.object(main, "UPLOAD_DIR", self.uploads)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(main.app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def post(self, name="lesson.pdf", data=None, content=b"sample document"):
        return self.client.post(
            "/api/v1/courses/generate",
            files={"file": (name, content, "application/octet-stream")},
            data=data if data is not None else {"course_id": "course-1"},
        )

    def test_upload_returns_all_components_and_removes_file(self):
        paths = []

        async def generate(path, course_id, requirement):
            paths.append(path)
            self.assertEqual(path.parent, self.uploads)
            self.assertEqual(path.read_bytes(), b"sample document")
            self.assertEqual((course_id, requirement), ("course-1", main.DOCUMENT_REQUIREMENT))
            return CourseGenerationResponse(course_id=course_id, **RESULTS)

        with patch.object(main.rag_service, "generate_course_materials", side_effect=generate):
            for name in ("../../lesson.pdf", "lesson.DOCX"):
                response = self.post(name)
                self.assertEqual(response.status_code, 200, response.text)
                initial_results = {key: RESULTS[key] for key in main.rag_service.INITIAL_CHAIN_NAMES}
                self.assertEqual(response.json(), {"course_id": "course-1", **initial_results})
                self.assertEqual(list(self.uploads.iterdir()), [])
        self.assertNotEqual(paths[0], paths[1])

    def test_failure_returns_generic_error_and_removes_file(self):
        with patch.object(main.rag_service, "generate_course_materials", new=AsyncMock(side_effect=RuntimeError("private credential"))):
            response = self.post()
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private credential", response.text)
        self.assertEqual(list(self.uploads.iterdir()), [])

    def test_rejects_unsupported_and_empty_files(self):
        with patch.object(main.rag_service, "generate_course_materials", new=AsyncMock()) as generate:
            self.assertEqual(self.post("lesson.txt").status_code, 415)
            self.assertEqual(self.post(content=b"").status_code, 400)
            generate.assert_not_awaited()
        self.assertEqual(list(self.uploads.iterdir()), [])

    def test_missing_and_blank_fields_are_rejected(self):
        for data in ({}, {"course_id": " "}, {"course_id": "a", "requirement": "HTML5"}):
            with self.subTest(data=data):
                self.assertEqual(self.post(data=data).status_code, 422)
        for data in ({"course_id": "a"}, {"course_id": "a", "requirement": " "}):
            self.assertEqual(self.client.post("/api/v1/courses/generate", data=data).status_code, 422)
        self.assertEqual(list(self.uploads.iterdir()), [])

    def test_requirement_only_generates_without_upload(self):
        result = CourseGenerationResponse(course_id="html5", **RESULTS)
        with patch.object(main.rag_service, "generate_course_materials", new=AsyncMock(return_value=result)) as generate:
            response = self.client.post("/api/v1/courses/generate", data={"course_id": "html5", "requirement": "Tôi muốn học HTML5"})
        self.assertEqual(response.status_code, 200, response.text)
        generate.assert_awaited_once_with(None, "html5", "Tôi muốn học HTML5")
        self.assertEqual(list(self.uploads.iterdir()), [])

    def test_health_checks_qdrant_and_reports_unavailable(self):
        qdrant = Mock()
        with patch.object(main.rag_service, "get_qdrant_client", return_value=qdrant):
            self.assertEqual(self.client.get("/health").status_code, 200)
            qdrant.get_collections.assert_called_once()
            qdrant.get_collections.side_effect = ConnectionError("private endpoint")
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private endpoint", response.text)

    def test_cors_preflight(self):
        response = self.client.options("/api/v1/courses/generate", headers={
            "Origin": main.CORS_ORIGINS[0], "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], main.CORS_ORIGINS[0])


if __name__ == "__main__":
    unittest.main()
