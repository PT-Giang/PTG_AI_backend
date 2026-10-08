from fastapi import APIRouter

from controllers.common import ChainGenerationRequest, generate_one_material
from rag_service import StudyQuestionSet


router = APIRouter(tags=["study questions"])


@router.post(
    "/api/v1/courses/{course_id}/materials/study-questions/generate",
    response_model=StudyQuestionSet,
)
async def generate_study_questions(course_id: str, payload: ChainGenerationRequest):
    return await generate_one_material(course_id, "study_questions", payload)
