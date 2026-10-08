# Implementation Plan: AI Backend Microservice (FastAPI & RAG Service)

## Overview
**Cập nhật yêu cầu đầu vào:** Có hai lựa chọn loại trừ nhau: gửi tài liệu (PDF/DOCX) hoặc gửi `requirement` độc lập như “Tôi muốn học HTML5”. `course_id` luôn bắt buộc. Không chấp nhận gửi cả hai. Chế độ yêu cầu dùng kiến thức mô hình, không chạy loader/embedding/Qdrant; chế độ tài liệu giữ RAG và tự tạo yêu cầu biên soạn học liệu tổng quát. Mô tả bên dưới ghi lại thiết kế RAG ban đầu; quy tắc đầu vào cập nhật này được ưu tiên.
**Cập nhật đầu ra:** Tạo ban đầu chỉ sinh `summary`, `study_guide`, `flashcards`, `study_questions`. Quiz được tạo riêng khi client gọi controller quiz.
Xây dựng dịch vụ AI Backend microservice phục vụ tạo trọn gói khóa học học tập từ tài liệu (.docx, .pdf) theo yêu cầu từ Spring Boot trong kiến trúc 3 tầng (Next.js -> Spring Boot -> FastAPI). Dịch vụ tiếp nhận file tài liệu và `course_id`, phân tách văn bản bằng mô hình hiện tại (`UnstructuredFileLoader`, `HuggingFaceEmbeddings`, `SemanticChunker`), index vào Qdrant phân tách theo `course_id`, sau đó kích hoạt 4 LLM Chains chạy song song (`asyncio.gather` + `ainvoke`) để sinh ra: Tóm tắt ý chính, Bộ Flashcard, Bộ câu hỏi học tập (trắc nghiệm/ghép từ kèm gợi ý), và Bộ đề kiểm tra trắc nghiệm chấm điểm; trả về kết quả JSON đồng bộ trực tiếp trong HTTP Response.

## Architecture Decisions
1. **Giữ nguyên 100% mô hình RAG hiện tại**:
   - Loader: `UnstructuredFileLoader(strategy="fast", languages=["vie", "eng"])` hỗ trợ cả DOCX và PDF.
   - Embedding: `HuggingFaceEmbeddings("intfloat/multilingual-e5-small")`.
   - Chunker: `SemanticChunker(embeddings=embeddings, breakpoint_threshold_amount=0.85)`.
   - Vector Store: `QdrantVectorStore` kết nối `QdrantClient`.
   - LLM: `ChatOpenAI(model="deepseek-v4-flash", temperature=0.2)`.
2. **Phân lập dữ liệu đa khóa học (Multi-tenancy by Course ID)**:
   - Thay vì `force_recreate=True` xóa sạch dữ liệu cũ như bản CLI cũ, ta gắn `split.metadata["course_id"] = course_id` vào mọi chunk và nạp vào Qdrant.
   - Khi tạo retriever để sinh bài, sử dụng bộ lọc Qdrant Filter theo `metadata.course_id` để chain AI chỉ đọc nội dung của đúng khóa học đó.
3. **Thực thi bất đồng bộ tối ưu thời gian phản hồi**:
   - Sử dụng `ainvoke` và `asyncio.gather` chạy đồng thời bốn chain ban đầu; quiz có endpoint/controller riêng để gọi theo yêu cầu.
4. **Quản lý file tạm an toàn**:
   - File upload lưu vào thư mục `temp_uploads/` và được đảm bảo xóa dọn dẹp trong khối `try...finally`.
5. **Bảo mật biến môi trường**:
   - Không đọc trực tiếp `.env`, tự động nạp qua `load_dotenv()` và `os.getenv`.

## Task List & Dependency Graph

```
Task 1: Cập nhật requirements.txt
    │
Task 2: Pydantic Schemas & Dữ liệu chuẩn (rag_service.py)
    │
Task 3: Pipeline Xử lý tài liệu & Qdrant Retriever theo course_id (rag_service.py)
    │
Task 4: Xây dựng 4 AI Chains & Orchestrator Async (rag_service.py)
    │
Task 5: Xây dựng FastAPI API Server & Endpoints (main.py)
    │
Task 6: Kiểm thử tích hợp tự động & thủ công E2E
```

---

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| SemanticChunker mất nhiều thời gian nếu file docx/pdf quá lớn | Medium | Giữ nguyên cấu hình `breakpoint_threshold_amount=0.85` của người dùng, sử dụng `strategy="fast"` của unstructured |
| LLM trả về JSON lỗi cú pháp hoặc kèm markdown fences | High | Dùng `JsonOutputParser` với `pydantic_object` và prompt định hướng rõ ràng `format_instructions` |
| Quên xóa file tạm khi request bị lỗi giữa chừng | Low | Sử dụng pattern `try...finally` ở tầng router trong `main.py` |
| Spring Boot bị timeout nếu LLM sinh chậm | High | Tối ưu bằng `asyncio.gather` để 4 chain chạy song song, rút ngắn thời gian sinh xuống còn 15-25s |

## Open Questions
- Không còn câu hỏi tồn đọng (đã chốt qua phiên interview: REST đồng bộ, phân loại câu hỏi trắc nghiệm/ghép từ, giữ nguyên pipeline RAG hiện tại).
