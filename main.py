import asyncio
import logging
import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from controllers import routers as material_routers
import rag_service
import chat_service
from chat_service import ChatRequest
from rag_service import InitialCourseMaterialsResponse


load_dotenv()
logger = logging.getLogger(__name__)
UPLOAD_DIR = Path(__file__).resolve().parent / "temp_uploads"
DOCUMENT_REQUIREMENT = "Tạo bộ học liệu tiếng Việt từ các nội dung chính trong tài liệu."
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


async def _process_upload(file: UploadFile, course_id: str, requirement: str):
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
        return await rag_service.generate_course_materials(path, course_id, requirement)
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
):
    if not course_id.strip():
        if file is not None:
            await file.close()
        raise HTTPException(status_code=422, detail="course_id must not be blank")
    if file is not None and requirement is not None:
        await file.close()
        raise HTTPException(status_code=422, detail="Choose either file or requirement, not both")
    if file is None:
        if requirement is None or not requirement.strip():
            raise HTTPException(status_code=422, detail="Provide either a file or a non-blank requirement")
        return await rag_service.generate_course_materials(None, course_id, requirement)
    # Khi request bị hủy, chờ worker ngừng dùng file rồi mới dọn file tạm.
    task = asyncio.create_task(_process_upload(file, course_id, DOCUMENT_REQUIREMENT))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        finally:
            raise
