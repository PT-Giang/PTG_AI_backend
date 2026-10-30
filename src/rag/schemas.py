from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import List, Literal
from uuid import UUID

from langchain_core.documents import Document
from pydantic import BaseModel, Field, model_validator


class Flashcard(BaseModel):
    front: str = Field(description="Mặt trước: Thuật ngữ, câu hỏi hoặc khái niệm chính")
    back: str = Field(description="Mặt sau: Giải thích ngắn gọn, định nghĩa hoặc câu trả lời")
    source_page: str = Field(description="Trích dẫn nguồn: Tên file và số trang")


class FlashcardSet(BaseModel):
    topic: str = Field(description="Chủ đề chính của bộ flashcard")
    cards: List[Flashcard] = Field(description="Danh sách các flashcard")


class QuizQuestion(BaseModel):
    type: Literal["multiple_choice", "short_answer"] = "multiple_choice"
    question: str = Field(description="Câu hỏi kiểm tra")
    options: List[str] | None = Field(default=None, description="4 phương án cho câu trắc nghiệm; null cho câu trả lời ngắn")
    correct_answer: str = Field(description="Đáp án đúng")
    explanation: str = Field(description="Giải thích đáp án dựa trên tài liệu")

    @model_validator(mode="after")
    def validate_question(self) -> "QuizQuestion":
        if self.type == "multiple_choice":
            if self.options is None or len(self.options) != 4 or len(set(self.options)) != 4:
                raise ValueError("multiple_choice requires four distinct options")
            if self.correct_answer not in self.options:
                raise ValueError("correct_answer must match an option exactly")
        elif self.options not in (None, []):
            raise ValueError("short_answer must not contain options")
        return self


class QuizSet(BaseModel):
    title: str = Field(description="Tiêu đề bài kiểm tra")
    questions: List[QuizQuestion] = Field(description="Danh sách câu hỏi trắc nghiệm")


class MatchingPair(BaseModel):
    term: str = Field(description="Thuật ngữ cần ghép")
    definition: str = Field(description="Định nghĩa đúng tương ứng với thuật ngữ")


class StudyQuestion(BaseModel):
    type: Literal["multiple_choice", "matching"] = Field(description="Dạng câu hỏi ôn tập")
    question: str = Field(description="Nội dung câu hỏi hoặc yêu cầu ghép cặp")
    hint: str = Field(description="Gợi ý giúp người học tự tìm câu trả lời")
    core_knowledge: str = Field(description="Kiến thức cốt lõi cần ghi nhớ sau câu hỏi")
    options: List[str] | None = Field(default=None, description="Đúng 4 phương án khác nhau cho multiple_choice; null cho matching")
    correct_answer: str | None = Field(default=None, description="Văn bản chính xác của phương án đúng; null cho matching")
    pairs: List[MatchingPair] | None = Field(default=None, description="Các cặp thuật ngữ - định nghĩa cho matching")

    @model_validator(mode="after")
    def validate_question_content(self) -> "StudyQuestion":
        if self.type == "multiple_choice":
            if self.options is None or len(self.options) != 4 or len(set(self.options)) != 4:
                raise ValueError("multiple_choice requires exactly four distinct options")
            if self.correct_answer not in self.options:
                raise ValueError("correct_answer must match an option exactly")
            if self.pairs is not None:
                raise ValueError("multiple_choice must not contain matching pairs")
        else:
            if not self.pairs:
                raise ValueError("matching requires at least one term-definition pair")
            if self.options is not None or self.correct_answer is not None:
                raise ValueError("matching must not contain multiple-choice answer fields")
        return self


class StudyQuestionSet(BaseModel):
    title: str = Field(description="Tiêu đề bộ câu hỏi ôn tập")
    questions: List[StudyQuestion] = Field(description="Câu hỏi trắc nghiệm và ghép cặp kèm gợi ý")


class PodcastEpisode(BaseModel):
    title: str = Field(min_length=1, max_length=240, description="Tiêu đề podcast")
    script: str = Field(min_length=1, max_length=50000, description="Kịch bản podcast dạng văn bản")
    duration_seconds: int | None = Field(default=None, ge=0, le=7200, description="Thời lượng ước tính nếu có")


class GenerationSource(BaseModel):
    source_type: Literal["document", "web"]
    title: str = Field(min_length=1, max_length=240)
    url: str | None = None
    excerpt: str = Field(default="", max_length=20000)
    retrieved_at: datetime
    is_mock: bool = False
    document_id: UUID | None = None


class CourseSummary(BaseModel):
    title: str = Field(min_length=1, max_length=120, description="Tên bộ học liệu thể hiện chủ đề chính")
    overview: str = Field(description="Tóm tắt tổng quan nội dung")
    key_points: List[str] = Field(description="Các ý chính và kiến thức trọng tâm")


class StudyGuide(BaseModel):
    title: str = Field(description="Tiêu đề mục hướng dẫn học tập")
    steps: List[str] = Field(description="Các chương Markdown theo mức độ kiến thức")
    tips: List[str] = Field(description="Lưu ý chuyên môn và lỗi dễ nhầm")


class CourseSummaryGenerationResponse(CourseSummary):
    sources: List[GenerationSource] = Field(default_factory=list)


class StudyGuideGenerationResponse(StudyGuide):
    sources: List[GenerationSource] = Field(default_factory=list)


class FlashcardSetGenerationResponse(FlashcardSet):
    sources: List[GenerationSource] = Field(default_factory=list)


class StudyQuestionSetGenerationResponse(StudyQuestionSet):
    sources: List[GenerationSource] = Field(default_factory=list)


class QuizSetGenerationResponse(QuizSet):
    sources: List[GenerationSource] = Field(default_factory=list)


class PodcastGenerationResponse(PodcastEpisode):
    sources: List[GenerationSource] = Field(default_factory=list)


class CourseGenerationResponse(BaseModel):
    course_id: str = Field(description="Mã khóa học sở hữu học liệu")
    summary: CourseSummary
    study_guide: StudyGuide
    flashcards: FlashcardSet
    study_questions: StudyQuestionSet
    quiz: QuizSet | None = None


class InitialCourseMaterialsResponse(BaseModel):
    course_id: str = Field(description="Mã khóa học sở hữu học liệu")
    summary: CourseSummary
    study_guide: StudyGuide
    flashcards: FlashcardSet
    study_questions: StudyQuestionSet
    sources: List[GenerationSource] = Field(default_factory=list)


INITIAL_CHAIN_NAMES = ("summary", "study_guide", "flashcards", "study_questions")
DOCUMENT_REQUIREMENT = "Tạo bộ học liệu tiếng Việt từ các nội dung chính trong tài liệu."


def build_source_metadata(documents: List[Document]) -> List[GenerationSource]:
    sources = []
    retrieved_at = datetime.now(timezone.utc)
    for document in documents[:30]:
        metadata = document.metadata or {}
        raw_title = str(metadata.get("source_name") or metadata.get("source") or "Tài liệu học tập")
        title = PurePosixPath(raw_title.replace("\\", "/")).name[:240] or "Tài liệu học tập"
        page = metadata.get("page")
        excerpt = document.page_content.strip()
        if page is not None:
            excerpt = f"Trang {page}: {excerpt}"
        source_url = metadata.get("source_url")
        source_type = "web" if isinstance(source_url, str) and source_url.startswith(("https://", "http://")) else "document"
        sources.append(GenerationSource(
            source_type=source_type, title=title, url=source_url if source_type == "web" else None,
            excerpt=excerpt[:20000], retrieved_at=retrieved_at, is_mock=False,
            document_id=metadata.get("document_id"),
        ))
    return sources
