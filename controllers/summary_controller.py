from fastapi import APIRouter

from controllers.common import ChainGenerationRequest, generate_one_material
from rag_service import CourseSummaryGenerationResponse


router = APIRouter(tags=["course summary"])


@router.post(
    "/api/v1/courses/{course_id}/materials/summary/generate",
    response_model=CourseSummaryGenerationResponse,
)
async def generate_summary(course_id: str, payload: ChainGenerationRequest):
    return await generate_one_material(course_id, "summary", payload)
