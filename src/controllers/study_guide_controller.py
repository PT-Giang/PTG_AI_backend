from fastapi import APIRouter
from pydantic import Field

from src.controllers.common import ChainGenerationRequest, generate_one_material
from src.rag.schemas import StudyGuideGenerationResponse


router = APIRouter(tags=["study guide"])


class StudyGuideRequest(ChainGenerationRequest):
    instruction: str = Field(min_length=1, max_length=100000)


@router.post(
    "/api/v1/courses/{course_id}/materials/study-guide/generate",
    response_model=StudyGuideGenerationResponse,
)
async def generate_study_guide(course_id: str, payload: StudyGuideRequest):
    return await generate_one_material(course_id, "study_guide", payload)
