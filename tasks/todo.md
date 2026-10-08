# Tasks: AI Backend Service

## Cập nhật đầu ra học tập
- `CourseGenerationResponse` hiện có thêm `study_guide`, gồm `title`, `steps` theo trình tự học và `tips` để ôn tập/tự kiểm tra.
- Endpoint tạo ban đầu chỉ sinh `summary`, `study_guide`, `flashcards`, `study_questions`.
- `quiz` chỉ được tạo khi gọi controller `/materials/quiz/generate` riêng.

## Cập nhật: hai lựa chọn đầu vào độc lập
- [x] API nhận `course_id` và đúng một trong `file` / `requirement`; gửi cả hai hoặc không có nội dung đều trả 422.
- [x] `requirement` là chủ đề/mục tiêu học độc lập; không cần tài liệu hay Qdrant. Bốn chain dùng kiến thức mô hình, có hướng dẫn không bịa trích dẫn tài liệu.
- [x] Upload chỉ gửi file và course_id; service tự dùng yêu cầu tạo học liệu tổng quát từ tài liệu.
- [x] Cập nhật test API, test service, script E2E tài liệu và README theo contract mới.
- [x] Kiểm tra cú pháp bằng py_compile và git diff --check.
- [x] Chạy lại bộ test runtime bằng Python global: `python -m unittest discover -v` → 27 tests passed. Xác nhận hai chế độ đầu vào, từ chối gửi cả hai, requirement độc lập không gọi pipeline tài liệu/Qdrant. LLM dùng giả lập.
- [x] Chạy lại E2E cả hai chế độ với LLM thật: requirement HTML5 HTTP 200 trong 16.0s (10 flashcards, 5 câu ôn tập, 5 quiz), không tạo collection; tài liệu Zero Trust HTTP 200 trong 39.0s (81 points, 6 flashcards, 5 câu ôn tập, 5 quiz). Đủ bốn thành phần, schema hợp lệ, file tạm/collection thử đã dọn. JSON: `test_results/requirement_html5.json`, `test_results/document_zero_trust.json`; báo cáo `test_results/e2e_report.json`.

## Phase 1: Foundation & Dependencies

### Task 1: Cập nhật file requirements.txt
**Description:** Bổ sung các thư viện cần thiết cho FastAPI và xử lý upload file.
**Acceptance criteria:**
- [x] `requirements.txt` có chứa `fastapi`, `uvicorn`, `python-multipart`.
- [x] Môi trường python `.venv` đã sẵn sàng các package này.
**Verification:**
- [x] Lệnh kiểm tra: `.\.venv\Scripts\python.exe -c "import fastapi, uvicorn, multipart; print('Dependencies OK')"` → `Dependencies OK`.
- [x] `.\.venv\Scripts\python.exe -m pip check` → `No broken requirements found.`
**Dependencies:** None
**Files likely touched:**
- `requirements.txt`
**Estimated scope:** XS (1 file)

### Checkpoint: Foundation
- [x] Dependencies cho FastAPI và upload file đã sẵn sàng trong `.venv`.

---

## Phase 2: RAG Service Core

### Task 2: Cập nhật Pydantic Schemas trong rag_service.py
**Description:** Định nghĩa đầy đủ các model dữ liệu cho toàn bộ thành phần học liệu: Flashcard, Quiz trắc nghiệm, Study Questions (trắc nghiệm ôn tập và ghép cặp từ - định nghĩa), Summary, và Response tổng hợp.
**Acceptance criteria:**
- [x] Có đầy đủ các model: `FlashcardSet`, `QuizSet`, `MatchingPair`, `StudyQuestion`, `StudyQuestionSet`, `CourseSummary`, `CourseGenerationResponse`.
- [x] `StudyQuestion` hỗ trợ cả dạng `multiple_choice` và `matching` kèm gợi ý và kiến thức cốt lõi.
- [x] `CourseGenerationResponse` tổng hợp các thành phần học liệu cùng `course_id`; hiện có thêm `study_guide`.
**Verification:**
- [x] Import: `.\.venv\Scripts\python.exe -c "from rag_service import CourseGenerationResponse; print('Schemas OK')"` → `Schemas OK`.
- [x] `.\.venv\Scripts\python.exe -m unittest discover -v` → 5 tests passed; kiểm tra dữ liệu mẫu, JSON round-trip, JSON Schema và từ chối câu hỏi không hợp lệ.
**Schema contract:** `course_id` là chuỗi; summary gồm `title`, `overview`, `key_points`; study questions gồm `title`, `questions`. Mỗi câu hỏi có `type`, `question`, `hint`, `core_knowledge`; `multiple_choice` yêu cầu 4 `options` khác nhau và `correct_answer` khớp một phương án; `matching` yêu cầu `pairs` gồm `term`, `definition`. Các trường dành cho dạng câu hỏi khác phải bỏ qua hoặc là `null`.
**Imports:** Thư viện RAG được import ở đầu `rag_service.py`; cần môi trường đã cài đủ dependencies để import module. Pipeline RAG chưa được kiểm thử ở Task 2.
**Dependencies:** Task 1
**Files likely touched:**
- `rag_service.py`
- `test_schemas.py`
**Estimated scope:** S (1 file)

### Task 3: Pipeline nạp tài liệu & Index Qdrant theo course_id
**Description:** Tái sử dụng `UnstructuredFileLoader`, `HuggingFaceEmbeddings`, `SemanticChunker(0.85)` và `QdrantVectorStore`. Gắn `metadata['course_id']` vào từng chunk, nạp vào collection Qdrant mà không xóa đè dữ liệu các khóa học khác, đồng thời tạo retriever có filter theo `course_id`.
**Acceptance criteria:**
- [x] Hàm `load_and_process_document(file_path, course_id)` xử lý `.docx` và `.pdf` bằng `UnstructuredFileLoader` (đã kiểm tra với mock loader).
- [x] Mỗi chunk có chứa `metadata["course_id"] = course_id`.
- [x] Dữ liệu được thêm vào Qdrant, không xóa collection hoặc dữ liệu cũ.
- [x] Hàm `get_course_retriever(course_id, k=6)` chỉ trả về chunks khớp đúng `course_id`.
**Verification:**
- [x] Kiểm tra với mock tài liệu, SemanticChunker thật, embedding giả lập cố định và Qdrant in-memory thật: phân lập hai khóa học, giữ metadata nguồn, nạp thêm, giới hạn k, từ chối input lỗi và tài liệu rỗng.
**Behavior:** Hàm nạp trả về danh sách chunks đã index. Nạp lại cùng tài liệu sẽ thêm chunks mới; chưa có chống trùng hoặc thay thế tài liệu. Dữ liệu cũ không có `course_id` sẽ không xuất hiện trong retriever theo khóa học. CLI yêu cầu mã khóa học và đường dẫn tài liệu; bỏ luồng re-index xóa toàn bộ collection.
**Verification environment:** Python global `C:\Users\PTG\AppData\Local\Programs\Python\Python310\python.exe`; `.venv` riêng của Codex chưa có thư viện RAG. Chưa chạy loader với PDF/DOCX thật, model HuggingFace thật hoặc Qdrant Docker ở Task 3.
- [x] Python global chạy `-m unittest discover -v`: 11 tests passed (6 pipeline/collection và 5 schema), bao gồm tạo collection, mở lại giữ dữ liệu và từ chối collection sai kích thước vector.
**Reference:** https://qdrant.tech/documentation/frameworks/langchain/
- [x] Kiểm thử tài liệu thật bằng Python global chạy `check_real_documents.py`: loader thật + `multilingual-e5-small` thật + Qdrant Docker 1.19.1. `200_tu_vung_tieng_anh.docx`: 397 chunks; `tai_lieu_nghien_cuu_zero_trust_tieng_viet.docx`: 81 chunks; tổng 478 points. Truy xuất k=3 đúng nguồn/course, truy vấn sau khi nạp cả hai vẫn phân lập khóa học, course không tồn tại trả rỗng. Thời gian 72.4 giây (không gồm import); collection thử đã xóa. Chưa kiểm thử PDF thật.
- [x] Kiểm thử server Docker thật bằng `python check_qdrant_server.py`: Qdrant 1.19.1; course-a trả 2 chunks, course-b trả 1, mã không tồn tại trả 0; k=1 đúng; nạp thêm giữ đủ 3 points. Collection thử riêng đã được xóa sau test. Loader/embedding vẫn giả lập; chưa kiểm tra trích xuất file và embedding thật.
**Dependencies:** Task 2
**Files likely touched:**
- `rag_service.py`
- `test_document_pipeline.py`
**Estimated scope:** M (1 file)

### Task 4: Xây dựng 4 AI Chains và Orchestrator Async
**Description:** Cấu hình các chain (`summary`, `study_guide`, `flashcards`, `study_questions`, `quiz`) sử dụng `ChatOpenAI(model="deepseek-v4-flash")` và `JsonOutputParser`. Hàm `generate_course_materials` chỉ sinh bốn mục ban đầu; quiz được gọi qua controller riêng.
**Acceptance criteria:**
- [x] Mỗi chain có prompt riêng, `JsonOutputParser` và bước kiểm tra Pydantic tương ứng.
- [x] Hàm `generate_course_materials(file_path, course_id, requirement)` điều phối luồng load doc -> Qdrant -> sinh bốn chains ban đầu song song -> trả về `InitialCourseMaterialsResponse`.
**Verification:**
- [x] Python global chạy `-m unittest discover -v`: 18 tests passed, gồm 7 tests Task 4; import module, tạo/invoke đủ 4 chain với LLM giả lập, từ chối JSON lỗi/schema lỗi, kiểm tra chạy song song, hủy các chain còn lại khi lỗi, từ chối requirement rỗng và dừng khi index lỗi.
**Contract:** `get_teacher_chains(retriever, chain_names)` tạo toàn bộ hoặc chỉ những chain được chọn. Mỗi chain nhận chuỗi yêu cầu và trả dict đã validate. Orchestrator chạy phần đồng bộ bằng `asyncio.to_thread`, gọi bốn `ainvoke` ban đầu qua `asyncio.gather`, rồi tạo `InitialCourseMaterialsResponse`. Controller quiz gọi riêng chain `quiz`. Lỗi được truyền lên caller; không trả kết quả thiếu thành phần, không xóa dữ liệu đã index.
**Verification scope:** Chưa gọi DeepSeek thật ở Task 4. Retrieval dùng k=6 chunks liên quan tới yêu cầu theo course_id, không đảm bảo bao phủ toàn bộ tài liệu dài.
**Dependencies:** Task 3
**Files likely touched:**
- `rag_service.py`
- `test_course_generation.py`
**Estimated scope:** M (1 file)

### Checkpoint: RAG Core
- [x] Module import thành công; pipeline nạp tài liệu đã kiểm tra thật ở Task 3, bốn chain/orchestrator đã kiểm tra bằng LLM giả lập ở Task 4. Kiểm thử LLM thật/E2E còn ở các bước tiếp theo.

---

## Phase 3: REST API Server & Endpoints

### Task 5: Xây dựng FastAPI app trong main.py
**Description:** Cấu hình ứng dụng FastAPI, CORS middleware, global exception handler, endpoint kiểm tra sức khỏe `GET /health`, và endpoint chính `POST /api/v1/courses/generate` nhận `UploadFile`, `course_id`, `requirement`. Xử lý lưu tạm và xóa file an toàn.
**Acceptance criteria:**
- [x] App FastAPI khởi tạo thành công với CORS (cấu hình qua `CORS_ORIGINS`).
- [x] `GET /health` kiểm tra kết nối tới Qdrant; trả 200 hoặc 503.
- [x] `POST /api/v1/courses/generate` xác thực đuôi file `.pdf`/`.docx` và file không rỗng, điều phối tới `rag_service.generate_course_materials`, tự dọn dẹp file tạm trong `finally`.
- [x] Trả về JSON đúng cấu trúc `CourseGenerationResponse`.
**Verification:**
- [x] Python global chạy `-m unittest discover -v`: 24 tests passed. Sáu test API import app và kiểm tra multipart thành công, dữ liệu thiếu/trắng, file sai đuôi/rỗng, health 200/503, CORS, lỗi 500 không lộ chi tiết và xóa file khi thành công/lỗi.
- [x] Kiểm tra cú pháp `main.py` và `test_api_routes.py` thành công.
**Contract:** Multipart `file`, `course_id`, `requirement`; response 200 có bốn mục ban đầu, không có quiz. Lỗi 415/400/422 cho input, 500 cho pipeline. File tạm tên riêng trong `temp_uploads/`; khi request bị hủy, chờ xử lý hoàn tất trước khi dọn file. Hướng dẫn chạy/cấu hình ở `README.md`.
**Verification scope:** Test HTTP qua FastAPI TestClient với pipeline và kết nối health giả lập; chưa chạy HTTP E2E gọi DeepSeek thật (Task 6).
**Dependencies:** Task 4
**Files likely touched:**
- `main.py`
- `test_api_routes.py`
- `.gitignore`
- `README.md`
**Estimated scope:** S (1 file)

### Task 6: Kiểm thử tích hợp tự động & End-to-End
**Description:** Viết kịch bản kiểm thử tích hợp (hoặc test script) gửi request multipart upload file mẫu trong `papers/` tới server, kiểm tra dữ liệu JSON phản hồi và xác nhận các trường dữ liệu.
**Acceptance criteria:**
- [x] Server phản hồi HTTP 200 qua HTTP thật tới Uvicorn.
- [x] JSON kết quả có đủ 4 thành phần: `summary`, `flashcards`, `study_questions`, `quiz`; validate bằng `CourseGenerationResponse`.
- [x] File tạm được xóa sạch sau khi xử lý xong.
**Verification:**
- [x] Python global chạy `python test_api.py` ngày 2026-09-22: tài liệu Zero Trust DOCX thật, embedding thật, Qdrant Docker 1.19.1, LLM theo cấu hình thật. HTTP 200; 81 points; 3 flashcards, 5 câu ôn tập (có cả multiple_choice/matching), 5 quiz; 48.7 giây không gồm import.
- [x] Sau kiểm thử: file upload đã xóa, collection thử đã xóa, server Uvicorn thử đã tắt. Sửa thứ tự đóng socket trong script để shutdown sạch trên Windows.
**Artifacts:** `test_results/course_response.json`, `test_results/e2e_report.json` (gitignored). Hướng dẫn chạy lại ở `README.md`.
**Limitations:** LLM chưa tuân thủ chính xác số lượng yêu cầu trong prompt (yêu cầu 2 câu ôn tập/3 quiz nhưng trả 5/5); số lượng chưa được kiểm tra cưỡng chế. E2E xác nhận luồng và schema, không xác nhận mọi phát biểu của LLM; chưa chạy PDF thật.
**Dependencies:** Task 5
**Files likely touched:**
- `test_api.py` (file kiểm thử)
**Estimated scope:** S (1 file)

### Checkpoint: Complete
- [x] Luồng AI Backend đã chạy E2E thành công và có API để Spring Boot kết nối. Chưa kiểm thử tích hợp từ ứng dụng Spring Boot thực tế.
