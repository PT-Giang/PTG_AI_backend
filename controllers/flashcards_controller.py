from fastapi import APIRouter

from controllers.common import ChainGenerationRequest, generate_one_material
from rag_service import FlashcardSet


router = APIRouter(tags=["flashcards"])


@router.post(
    "/api/v1/courses/{course_id}/materials/flashcards/generate",
    response_model=FlashcardSet,
)
async def generate_flashcards(course_id: str, payload: ChainGenerationRequest):
    return await generate_one_material(course_id, "flashcards", payload)
