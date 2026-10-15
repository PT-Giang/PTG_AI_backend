from fastapi import APIRouter

from controllers.common import ChainGenerationRequest, generate_one_material
from rag_service import PodcastGenerationResponse


router = APIRouter(tags=["podcast"])


@router.post(
    "/api/v1/courses/{course_id}/materials/podcast/generate",
    response_model=PodcastGenerationResponse,
)
async def generate_podcast(course_id: str, payload: ChainGenerationRequest):
    return await generate_one_material(course_id, "podcast", payload)
