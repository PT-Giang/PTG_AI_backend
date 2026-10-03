"""Explicit real E2E test: python test_api.py. Calls the configured LLM API."""
import json
import os
import socket
import threading
import time
from pathlib import Path
from uuid import uuid4

import httpx
import uvicorn
from dotenv import load_dotenv
from qdrant_client import QdrantClient

import main
from rag_service import CourseGenerationResponse


def run_e2e():
    load_dotenv()
    if not (os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise RuntimeError("Configure DEEPSEEK_API_KEY or OPENAI_API_KEY before running E2E")
    root = Path(__file__).resolve().parent
    document = root / "papers" / "tai_lieu_nghien_cuu_zero_trust_tieng_viet.docx"
    if not document.is_file():
        raise FileNotFoundError(document)
    collection = "e2e_test_" + uuid4().hex
    course_id = "e2e-course-" + uuid4().hex
    previous_collection = os.environ.get("QDRANT_COLLECTION_NAME")
    os.environ["QDRANT_COLLECTION_NAME"] = collection
    qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"), timeout=30)
    server = None
    listener = None
    thread = None
    owned_collection = False
    uploads_before = set(main.UPLOAD_DIR.glob("*"))
    report = {"document": document.name, "course_id": course_id, "success": False}
    started = time.monotonic()
    output = root / "test_results"
    output.mkdir(exist_ok=True)
    try:
        report["qdrant_version"] = qdrant.info().version
        assert not qdrant.collection_exists(collection), "Test collection unexpectedly exists"
        owned_collection = True
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        config = uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        deadline = time.monotonic() + 30
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("Uvicorn did not start")
            time.sleep(0.1)
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=600, trust_env=False) as client:
            health = client.get("/health")
            assert health.status_code == 200, f"Health returned {health.status_code}"
            print("Testing requirement-only mode: HTML5...", flush=True)
            topic_id = "html5-" + uuid4().hex
            topic_started = time.monotonic()
            topic_response = client.post(
                "/api/v1/courses/generate",
                data={"course_id": topic_id, "requirement": "Tôi muốn học HTML5 từ cơ bản, dành cho người mới bắt đầu."},
            )
            report["requirement_mode"] = {"http_status": topic_response.status_code}
            assert topic_response.status_code == 200, f"Requirement mode returned HTTP {topic_response.status_code}"
            topic = CourseGenerationResponse.model_validate(topic_response.json())
            assert topic.course_id == topic_id
            assert topic.summary.overview and topic.summary.key_points
            assert topic.flashcards.cards and topic.study_questions.questions and topic.quiz.questions
            assert {q.type for q in topic.study_questions.questions} == {"multiple_choice", "matching"}
            assert all(len(q.options) == 4 and q.correct_answer in q.options for q in topic.quiz.questions)
            assert all(card.source_page == "Kiến thức tổng hợp" for card in topic.flashcards.cards)
            assert not qdrant.collection_exists(collection), "Requirement mode unexpectedly indexed data"
            assert set(main.UPLOAD_DIR.glob("*")) == uploads_before
            (output / "requirement_html5.json").write_text(topic.model_dump_json(indent=2), encoding="utf-8")
            report["requirement_mode"].update(
                success=True, flashcards=len(topic.flashcards.cards), study_questions=len(topic.study_questions.questions),
                quiz=len(topic.quiz.questions), elapsed_seconds=round(time.monotonic() - topic_started, 1),
                no_collection_created=True,
            )
            print("PASS requirement-only HTTP 200; JSON saved; no collection created", flush=True)
            print("PASS HTTP health; sending real document to generation endpoint...", flush=True)
            document_started = time.monotonic()
            with document.open("rb") as file:
                response = client.post(
                    "/api/v1/courses/generate",
                    data={"course_id": course_id},
                    files={"file": (document.name, file, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
                )
            report["http_status"] = response.status_code
            print("Generation HTTP status:", response.status_code, flush=True)
            assert response.status_code == 200, f"Generation returned HTTP {response.status_code}"
            result = CourseGenerationResponse.model_validate(response.json())
            assert result.course_id == course_id
            assert result.summary.overview and result.summary.key_points
            assert result.flashcards.cards and result.study_questions.questions and result.quiz.questions
            assert {q.type for q in result.study_questions.questions} == {"multiple_choice", "matching"}
            assert all(len(q.options) == 4 and q.correct_answer in q.options for q in result.quiz.questions)
            (output / "course_response.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
            (output / "document_zero_trust.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
            report["document_seconds"] = round(time.monotonic() - document_started, 1)
            report.update(flashcards=len(result.flashcards.cards), study_questions=len(result.study_questions.questions), quiz=len(result.quiz.questions))
            report["indexed_points"] = qdrant.count(collection, exact=True).count
            assert report["indexed_points"] > 0
            assert set(main.UPLOAD_DIR.glob("*")) == uploads_before, "Temporary upload was not cleaned"
            report["success"] = True
            print("PASS schema, all four components, Qdrant indexing and upload cleanup", flush=True)
    finally:
        try:
            if server is not None:
                server.should_exit = True
            if thread is not None:
                thread.join(timeout=60)
                if thread.is_alive():
                    raise RuntimeError("Test server still running; keeping its collection for diagnosis")
            if listener is not None:
                listener.close()
            report["uploads_cleaned"] = set(main.UPLOAD_DIR.glob("*")) == uploads_before
            if owned_collection and qdrant.collection_exists(collection):
                qdrant.delete_collection(collection)
                assert not qdrant.collection_exists(collection)
                print("Temporary Qdrant collection removed", flush=True)
            report["collection_cleaned"] = owned_collection
        finally:
            qdrant.close()
            report["elapsed_seconds"] = round(time.monotonic() - started, 1)
            (output / "e2e_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            if previous_collection is None:
                os.environ.pop("QDRANT_COLLECTION_NAME", None)
            else:
                os.environ["QDRANT_COLLECTION_NAME"] = previous_collection


if __name__ == "__main__":
    run_e2e()
