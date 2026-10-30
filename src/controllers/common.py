from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.rag import service as rag_service


class ChainGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    material_source: Literal["document", "requirement"]
    instruction: str = Field(min_length=1, max_length=8000)
    document_id: UUID | None = None

    @field_validator("instruction")
    @classmethod
    def instruction_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instruction must not be blank")
        return value


async def generate_one_material(course_id: str, chain_name: str, payload: ChainGenerationRequest):
    return await rag_service.generate_one_material(course_id, chain_name, payload)
