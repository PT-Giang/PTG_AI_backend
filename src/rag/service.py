import asyncio
from pathlib import Path
from typing import List
from uuid import UUID
from fastapi import HTTPException

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.rag.llm import format_docs, get_teacher_chains
from src.rag.schemas import (
    DOCUMENT_REQUIREMENT, INITIAL_CHAIN_NAMES, GenerationSource,
    InitialCourseMaterialsResponse, PodcastGenerationResponse,
    CourseSummaryGenerationResponse, StudyGuideGenerationResponse,
    FlashcardSetGenerationResponse, StudyQuestionSetGenerationResponse,
    QuizSetGenerationResponse, build_source_metadata,
)
from src.rag.vector_store import (
    get_course_retriever, get_qdrant_client, get_vector_store,
    index_text_document, load_and_process_document, validate_course_id,
)


async def generate_course_materials(
    file_path: str | Path | None,
    course_id: str,
    requirement: str,
    source_name: str | None = None,
    document_id: UUID | None = None,
) -> InitialCourseMaterialsResponse:
    """Index tài liệu nếu có rồi sinh bốn phần ban đầu; quiz được tạo riêng."""
    validate_course_id(course_id)
    if not isinstance(requirement, str) or not requirement.strip():
        raise ValueError("requirement must be a non-empty string")
    documents: List[Document] = []
    retriever = None
    if file_path is not None:
        if document_id is not None:
            await asyncio.to_thread(load_and_process_document, file_path, course_id, source_name, document_id)
        elif source_name is None:
            await asyncio.to_thread(load_and_process_document, file_path, course_id)
        else:
            await asyncio.to_thread(load_and_process_document, file_path, course_id, source_name)
        retriever = await asyncio.to_thread(get_course_retriever, course_id, 6, document_id) if document_id else await asyncio.to_thread(get_course_retriever, course_id)
        documents = await asyncio.to_thread(retriever.invoke, requirement)
        retriever = RunnableLambda(lambda _: documents)
    chains = await asyncio.to_thread(get_teacher_chains, retriever, INITIAL_CHAIN_NAMES)
    chains = {name: chains[name] for name in INITIAL_CHAIN_NAMES}
    tasks = [asyncio.create_task(chain.ainvoke(requirement)) for chain in chains.values()]
    try:
        results = await asyncio.gather(*tasks)
    except (Exception, asyncio.CancelledError):
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return InitialCourseMaterialsResponse(
        course_id=course_id,
        **dict(zip(chains.keys(), results)),
        sources=build_source_metadata(documents),
    )


async def generate_course_materials_from_text(
    course_id: str, document_id: UUID, source_name: str, text: str,
) -> InitialCourseMaterialsResponse:
    """Index ghi chú từ client rồi sinh học liệu từ nội dung đó."""
    if not source_name.strip() or not text.strip():
        raise ValueError("source_name and text must be non-empty")
    await asyncio.to_thread(index_text_document, course_id, document_id, source_name, text)
    retriever = await asyncio.to_thread(get_course_retriever, course_id, 6, document_id)
    documents = await asyncio.to_thread(retriever.invoke, DOCUMENT_REQUIREMENT)
    chains = await asyncio.to_thread(
        get_teacher_chains, RunnableLambda(lambda _: documents), INITIAL_CHAIN_NAMES
    )
    chains = {name: chains[name] for name in INITIAL_CHAIN_NAMES}
    tasks = [asyncio.create_task(chain.ainvoke(DOCUMENT_REQUIREMENT)) for chain in chains.values()]
    try:
        results = await asyncio.gather(*tasks)
    except (Exception, asyncio.CancelledError):
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return InitialCourseMaterialsResponse(
        course_id=course_id,
        **dict(zip(chains.keys(), results)),
        sources=build_source_metadata(documents),
    )


async def generate_one_material(course_id: str, chain_name: str, payload):
    validate_course_id(course_id)
    retriever = None
    document_id = getattr(payload, "document_id", None)
    if payload.material_source == "document":
        retriever = await asyncio.to_thread(get_course_retriever, course_id, 6, document_id)
        documents = await asyncio.to_thread(retriever.invoke, payload.instruction)
        if not documents or not any(doc.page_content.strip() for doc in documents):
            raise HTTPException(status_code=422, detail="Không tìm thấy nội dung của tài liệu đã chọn. Hãy nạp lại tài liệu trước khi tạo học liệu.")
        retriever = RunnableLambda(lambda _: documents)
    chains = await asyncio.to_thread(get_teacher_chains, retriever, [chain_name])
    result = await chains[chain_name].ainvoke(payload.instruction)
    response_types = {
        "summary": CourseSummaryGenerationResponse,
        "study_guide": StudyGuideGenerationResponse,
        "flashcards": FlashcardSetGenerationResponse,
        "study_questions": StudyQuestionSetGenerationResponse,
        "quiz": QuizSetGenerationResponse,
        "podcast": PodcastGenerationResponse,
    }
    response_type = response_types[chain_name]
    if payload.material_source == "document":
        return response_type(**result, sources=build_source_metadata(documents))
    return response_type(**result)
