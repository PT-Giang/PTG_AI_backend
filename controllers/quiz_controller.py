from fastapi import APIRouter
from pydantic import Field

from controllers.common import ChainGenerationRequest, generate_one_material
from rag_service import QuizSet


router = APIRouter(tags=["quiz"])


class QuizGenerationRequest(ChainGenerationRequest):
    question_count: int = Field(default=5, ge=1, le=15)


@router.post(
    "/api/v1/courses/{course_id}/materials/quiz/generate",
    response_model=QuizSet,
)
async def generate_quiz(course_id: str, payload: QuizGenerationRequest):
    request_with_count = payload.model_copy(update={
        "instruction": (
            f"{payload.instruction}\n\nTạo chính xác {payload.question_count} câu hỏi quiz. "\
            "Không tạo nhiều hay ít hơn số lượng câu hỏi này."   
            
        ),
    })
    result = await generate_one_material(course_id, "quiz", request_with_count)
    result["questions"] = result["questions"][:payload.question_count]
    return result
