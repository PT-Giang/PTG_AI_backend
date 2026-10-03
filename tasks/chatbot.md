# Chatbot học tập — FastAPI

## Phạm vi đã chốt
Client → Spring Boot → FastAPI → LLM; trả lời hiện dần qua SSE.
Spring Boot xác thực, kiểm tra quyền khóa học và sở hữu lịch sử. FastAPI không lưu hội thoại.
Repo hiện tại chỉ triển khai FastAPI và contract để Spring Boot tích hợp; chưa sửa UI/backend Java.

## API
`POST /api/v1/chat` nhận JSON:
- `message`: nội dung hiện tại, 1–8000 ký tự, không rỗng/trắng.
- `history`: tối đa 20 tin nhắn gần nhất theo thứ tự cũ → mới, chỉ role user/assistant, mỗi tin tối đa 8000 ký tự. Không chứa lại message hiện tại.
- `course_id`: tùy chọn. Không có: chat tổng quát, không gọi Qdrant.
- `course_materials`: tùy chọn, cấu trúc CourseGenerationResponse do Spring Boot gửi từ dữ liệu lưu trữ, course_id phải khớp. Dùng cho khóa học sinh từ yêu cầu; không gọi Qdrant khi đã có học liệu này.

Có course_id nhưng không có course_materials: truy xuất tài liệu bằng retriever lọc course_id, lấy ngữ cảnh từ câu hỏi và các lượt user gần nhất. Nếu không tìm thấy nội dung: nói rõ thiếu dữ liệu, không giả vờ biết khóa học.
Ngữ cảnh khóa học và lịch sử là dữ liệu, không được thay đổi chỉ dẫn hệ thống. Chỉ trích nguồn có trong dữ liệu nhận được.

SSE `text/event-stream`: `start` → `delta` (content) → `done` (full content).
Lỗi trong stream: `error` với thông báo chung rồi kết thúc, không có done; Spring Boot phải đánh dấu câu trả lời dở dang, không coi HTTP 200 là thành công nếu thiếu done.
Invalid body: HTTP 422 trước khi stream. Streaming không lưu lịch sử và không index tài liệu mới. Retriever hiện tại có thể khởi tạo collection rỗng nếu chưa tồn tại.
Hủy kết nối phải đóng stream LLM. Giới hạn tổng history 32000 ký tự, context 48000 ký tự; không tự cắt message hiện tại.

## Triển khai / xác minh
1. chat_service.py: schemas, messages, retrieval, async LLM streaming, SSE.
2. main.py: route StreamingResponse; không thay đổi các API tạo học liệu.
3. test_chat.py: validation, history, tổng quát không gọi retrieval, course filter/context, SSE delta/done/error, hủy stream.
4. README.md: contract, ví dụ và trách nhiệm Spring Boot chuyển tiếp SSE.

Chạy: `python -m uvicorn main:app --port 8000`.
Kiểm thử: `python -m unittest discover -v` bằng Python global 3.10.
Style: imports đầu file, Pydantic v2, async cho LLM; việc khởi tạo retriever đồng bộ dùng asyncio.to_thread.
Chỉ gọi LLM thật khi cần kiểm tra thực tế; test tự động dùng giả lập.

## Kết quả triển khai
- [x] API `/api/v1/chat` nhận JSON và trả SSE; chat tổng quát, khóa học tài liệu, khóa học gửi học liệu đều có nhánh xử lý.
- [x] Kiểm tra giới hạn dữ liệu và role; không chấp nhận system message từ caller.
- [x] Stream delta từng phần, done đầy đủ, lỗi chung qua error; đóng upstream khi đóng response.
- [x] Python global chạy `python -m unittest discover -v`: 35 tests passed (8 chatbot + 27 test cũ).
- [x] Kiểm tra cú pháp thành công; README có contract tích hợp Spring Boot.
- [ ] Chạy streaming với LLM thật và kiểm tra chuyển tiếp qua Spring Boot/client (chưa nằm trong phần đã kiểm thử).
