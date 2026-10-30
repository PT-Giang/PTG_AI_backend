import asyncio
import logging
import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from uuid import UUID

from src.controllers import routers as material_routers
from src import chat_service
from src.rag import service as rag_service
from src.chat_service import ChatRequest
from src.rag.schemas import DOCUMENT_REQUIREMENT, InitialCourseMaterialsResponse


load_dotenv()
logger = logging.getLogger(__name__)
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "temp_uploads"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*",).split(",")
    if origin.strip()
]

app = FastAPI(title="AI Course Backend", version="1.0.0")
for material_router in material_routers:
    app.include_router(material_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.exception_handler(RequestValidationError)
async def request_validation_error(request: Request, exc: RequestValidationError):
    """Log schema errors without echoing user document text or other submitted values."""
    safe_errors = [
        {
            "loc": [str(part) for part in error.get("loc", ())],
            "msg": str(error.get("msg", "Invalid value")),
            "type": str(error.get("type", "validation_error")),
        }
        for error in exc.errors()
    ]
    logger.warning("Invalid request payload for %s: %s", request.url.path, safe_errors)
    return JSONResponse(status_code=422, content={"detail": safe_errors})


class IndexDocumentRequest(BaseModel):
    document_id: UUID
    source_name: str = Field(min_length=1, max_length=240)
    text: str = Field(min_length=1, max_length=1_500_000)


@app.post("/api/v1/courses/{course_id}/documents/index")
async def index_document(course_id: str, payload: IndexDocumentRequest):
    if not payload.source_name.strip() or not payload.text.strip():
        raise HTTPException(status_code=422, detail="source_name and text must not be blank")
    try:
        count = await asyncio.to_thread(
            rag_service.index_text_document,
            course_id,
            payload.document_id,
            payload.source_name,
            payload.text,
        )
    except ValueError as error:
        logger.warning(
            "Document indexing rejected for document %s: %s",
            payload.document_id,
            str(error)[:240],
        )
        raise HTTPException(status_code=422, detail="Document could not be indexed") from error
    return {"course_id": course_id, "document_id": str(payload.document_id), "chunk_count": count}


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    logger.error("Unhandled error on %s", request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
def health():
    """Kiểm tra kết nối Qdrant mà không tải embedding hoặc gọi LLM."""
    try:
        rag_service.get_qdrant_client().get_collections()
    except Exception:
        logger.warning("Qdrant health check failed", exc_info=True)
        return JSONResponse(status_code=503, content={"status": "unavailable", "qdrant": "disconnected"})
    return {"status": "ok", "qdrant": "connected"}


@app.post("/api/v1/chat", response_class=StreamingResponse,
          responses={200: {"content": {"text/event-stream": {}}}})
async def chat(payload: ChatRequest):
    return StreamingResponse(
        chat_service.stream_chat(payload),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _save_upload(file: UploadFile, path: Path):
    with path.open("wb") as destination:
        shutil.copyfileobj(file.file, destination, length=1024 * 1024)


async def _process_upload(file: UploadFile, course_id: str, requirement: str, document_id: UUID | None = None):
    path = None
    try:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".pdf", ".docx"}:
            raise HTTPException(status_code=415, detail="Only .pdf and .docx files are supported")
        if not course_id.strip() or not requirement.strip():
            raise HTTPException(status_code=422, detail="course_id and requirement must not be blank")
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        # Tên riêng cho từng request, không dùng đường dẫn do client gửi lên.
        with NamedTemporaryFile(dir=UPLOAD_DIR, suffix=suffix, delete=False) as temp:
            path = Path(temp.name)
        await asyncio.to_thread(_save_upload, file, path)
        if path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if document_id is not None:
            return await rag_service.generate_course_materials(path, course_id, requirement, file.filename, document_id)
        return await rag_service.generate_course_materials(path, course_id, requirement, file.filename)
    finally:
        try:
            if path is not None:
                path.unlink(missing_ok=True)
        finally:
            await file.close()


@app.post("/api/v1/courses/generate", response_model=InitialCourseMaterialsResponse)
async def generate_course(
    course_id: Annotated[str, Form(min_length=1)],
    file: Annotated[UploadFile | None, File(description="Chọn tài liệu PDF/DOCX, không gửi requirement")] = None,
    requirement: Annotated[str | None, Form(description="Hôm nay bạn muốn học gì? Không gửi file")] = None,
    content: Annotated[str | None, Form(description="Ghi chú hoặc thuật ngữ dùng làm nguồn học liệu")] = None,
    document_id: Annotated[UUID | None, Form()] = None,
    source_name: Annotated[str | None, Form()] = None,
):
    if not course_id.strip():
        if file is not None:
            await file.close()
        raise HTTPException(status_code=422, detail="course_id must not be blank")
    supplied = sum(value is not None for value in (file, requirement, content))
    if supplied != 1:
        if file is not None:
            await file.close()
        raise HTTPException(status_code=422, detail="Choose exactly one of file, requirement, or content")
    if content is not None:
        if document_id is None or source_name is None or not content.strip() or not source_name.strip():
            raise HTTPException(status_code=422, detail="content requires document_id and source_name")
        return await rag_service.generate_course_materials_from_text(course_id, document_id, source_name, content)
    if file is None:
        if requirement is None or not requirement.strip():
            raise HTTPException(status_code=422, detail="Provide either a file or a non-blank requirement")
        return await rag_service.generate_course_materials(None, course_id, requirement)
    # Khi request bị hủy, chờ worker ngừng dùng file rồi mới dọn file tạm.
    task = asyncio.create_task(_process_upload(file, course_id, DOCUMENT_REQUIREMENT, document_id))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        finally:
            raise
