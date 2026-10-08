from fastapi import APIRouter

from controllers.common import ChainGenerationRequest, generate_one_material
from rag_service import StudyGuide


router = APIRouter(tags=["study guide"])


@router.post(
    "/api/v1/courses/{course_id}/materials/study-guide/generate",
    response_model=StudyGuide,
)
async def generate_study_guide(course_id: str, payload: ChainGenerationRequest):
    return await generate_one_material(course_id, "study_guide", payload)
