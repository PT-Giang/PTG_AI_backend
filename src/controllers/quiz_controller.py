from fastapi import APIRouter, HTTPException
from pydantic import Field, model_validator

from src.controllers.common import ChainGenerationRequest, generate_one_material
from src.rag.schemas import QuizSetGenerationResponse


router = APIRouter(tags=["quiz"])


class QuizGenerationRequest(ChainGenerationRequest):
    question_count: int = Field(default=5, ge=1, le=15)
    choice_count: int = Field(default=5, ge=0, le=15)
    short_answer_count: int = Field(default=0, ge=0, le=15)

    @model_validator(mode="before")
    @classmethod
    def infer_legacy_counts(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            if "choice_count" not in value and "short_answer_count" not in value:
                value["choice_count"] = value.get("question_count", 5)
                value["short_answer_count"] = 0
            elif "choice_count" not in value:
                value["choice_count"] = value.get("question_count", 5) - value["short_answer_count"]
            elif "short_answer_count" not in value:
                value["short_answer_count"] = value.get("question_count", 5) - value["choice_count"]
        return value

    @model_validator(mode="after")
    def counts_must_match_total(self):
        if self.choice_count + self.short_answer_count != self.question_count:
            raise ValueError("question_count must equal choice_count + short_answer_count")
        return self


@router.post(
    "/api/v1/courses/{course_id}/materials/quiz/generate",
    response_model=QuizSetGenerationResponse,
)
async def generate_quiz(course_id: str, payload: QuizGenerationRequest):
    request_with_count = payload.model_copy(update={
        "instruction": (
            f"{payload.instruction}\n\nTạo chính xác {payload.question_count} câu hỏi quiz. "\
            f"Trong đó có chính xác {payload.choice_count} câu multiple_choice và "
            f"{payload.short_answer_count} câu short_answer. Không tạo nhiều hay ít hơn."
        ),
    })
    result = QuizSetGenerationResponse.model_validate(await generate_one_material(course_id, "quiz", request_with_count))
    
    # 1. Lấy danh sách questions từ result object
    questions = result.questions

    # 2. Đảm bảo hỗ trợ cả trường hợp question là Pydantic Model lẫn dict
    def get_q_type(q):
        return getattr(q, "type", None) if hasattr(q, "type") else q.get("type", "multiple_choice")

    choices = [q for q in questions if get_q_type(q) == "multiple_choice"]
    short_answers = [q for q in questions if get_q_type(q) == "short_answer"]

    # 3. Kiểm tra số lượng câu hỏi
    if len(choices) < payload.choice_count or len(short_answers) < payload.short_answer_count:
        raise HTTPException(status_code=502, detail="AI chưa tạo đủ số lượng từng dạng câu hỏi yêu cầu.")

    # 4. Gán lại danh sách questions thông qua attribute (hoặc dùng model_copy)
    result.questions = choices[:payload.choice_count] + short_answers[:payload.short_answer_count]
    
    return result
