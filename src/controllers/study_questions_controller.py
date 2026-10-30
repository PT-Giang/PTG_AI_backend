from fastapi import APIRouter

from src.controllers.common import ChainGenerationRequest, generate_one_material
from src.rag.schemas import StudyQuestionSetGenerationResponse


router = APIRouter(tags=["study questions"])


@router.post(
    "/api/v1/courses/{course_id}/materials/study-questions/generate",
    response_model=StudyQuestionSetGenerationResponse,
)
async def generate_study_questions(course_id: str, payload: ChainGenerationRequest):
    return await generate_one_material(course_id, "study_questions", payload)
