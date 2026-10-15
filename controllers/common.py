import asyncio
from typing import Literal

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ConfigDict, Field, field_validator

import rag_service
from rag_service import build_source_metadata


class ChainGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    material_source: Literal["document", "requirement"]
    instruction: str = Field(min_length=1, max_length=8000)

    @field_validator("instruction")
    @classmethod
    def instruction_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instruction must not be blank")
        return value


async def generate_one_material(course_id: str, chain_name: str, payload: ChainGenerationRequest):
    rag_service._validate_course_id(course_id)
    documents = []
    retriever = None
    if payload.material_source == "document":
        retriever = await asyncio.to_thread(rag_service.get_course_retriever, course_id)
        documents = await asyncio.to_thread(retriever.invoke, payload.instruction)
        retriever = RunnableLambda(lambda _: documents)
    chains = await asyncio.to_thread(rag_service.get_teacher_chains, retriever, [chain_name])
    result = await chains[chain_name].ainvoke(payload.instruction)
    result["sources"] = [source.model_dump(mode="json") for source in build_source_metadata(documents)]
    return result
