from fastapi import APIRouter

from src.controllers.common import ChainGenerationRequest, generate_one_material
from src.rag.schemas import FlashcardSetGenerationResponse


router = APIRouter(tags=["flashcards"])


@router.post(
    "/api/v1/courses/{course_id}/materials/flashcards/generate",
    response_model=FlashcardSetGenerationResponse,
)
async def generate_flashcards(course_id: str, payload: ChainGenerationRequest):
    return await generate_one_material(course_id, "flashcards", payload)
