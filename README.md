# AI Course Backend

## Chạy API

Dùng môi trường Python đã cài `requirements.txt` (hiện dự án dùng Python global 3.10):

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Swagger UI: `http://127.0.0.1:8000/docs`.

Biến môi trường được nạp bằng `load_dotenv()`:

- `QDRANT_URL`: mặc định `http://localhost:6333`.
- `QDRANT_COLLECTION_NAME`: mặc định `papers_collection`.
- `DEEPSEEK_API_KEY` (hoặc `OPENAI_API_KEY`).
- `DEEPSEEK_API_BASE`: mặc định `https://api.deepseek.com`.
- `CORS_ORIGINS`: danh sách origin cách nhau bằng dấu phẩy; mặc định `http://localhost:3000`.

## HTTP API

`GET /health` kiểm tra Qdrant: trả 200 với `{"status":"ok","qdrant":"connected"}`, hoặc 503 khi không kết nối được. Endpoint này không kiểm tra LLM.

`POST /api/v1/courses/generate` yêu cầu `course_id` và **đúng một** trong hai lựa chọn:

- `file`: upload PDF/DOCX, AI tạo học liệu từ nội dung tài liệu. Không gửi `requirement`.
- `requirement`: yêu cầu học độc lập, ví dụ “Tôi muốn học HTML5”. AI dùng kiến thức của mô hình để tạo chủ đề và bộ học liệu; không cần file, không index/truy xuất Qdrant. Không gửi `file`.

Nếu gửi cả hai hoặc không gửi lựa chọn nào, API trả 422. `requirement` dùng form field; có thể gửi `application/x-www-form-urlencoded` khi không có file. Hai ví dụ PowerShell:

```powershell
curl.exe http://127.0.0.1:8000/api/v1/courses/generate -F "file=@papers/200_tu_vung_tieng_anh.docx" -F "course_id=english-1"

curl.exe http://127.0.0.1:8000/api/v1/courses/generate -F "course_id=html5-1" -F "requirement=Tôi muốn học HTML5 từ cơ bản"
```

Cả hai cách gọi LLM thật; chỉ chế độ tài liệu thêm dữ liệu vào Qdrant. Kết quả HTTP 200 gồm `course_id`, `summary`, `flashcards`, `study_questions`, `quiz`. Với chế độ yêu cầu, flashcard được hướng dẫn ghi nguồn `Kiến thức tổng hợp`, không bịa tên tài liệu/số trang. API đợi hoàn tất cả bốn thành phần trước khi trả kết quả; phía gọi cần timeout đủ cho xử lý tài liệu và LLM.

Lỗi: 415 khi đuôi file không phải PDF/DOCX, 400 khi file rỗng, 422 khi thiếu hoặc bỏ trắng trường bắt buộc, 500 khi pipeline lỗi. Đuôi file không đảm bảo nội dung hợp lệ; lỗi giải mã tài liệu hiện trả 500. Chi tiết lỗi nội bộ nằm trong log server.

File tạm nằm trong `temp_uploads/` và được xóa khi xử lý kết thúc, kể cả khi lỗi. Khi request bị hủy, server chờ tác vụ xử lý kết thúc trước khi dọn file để tránh xóa file đang được worker đọc. Nạp lại cùng tài liệu sẽ thêm chunks mới.

## Chatbot streaming qua Spring Boot

Luồng: Client → Spring Boot → FastAPI → LLM. Spring Boot xác thực người dùng, kiểm tra quyền khóa học, lấy lịch sử của đúng cuộc hội thoại, gọi FastAPI và chuyển tiếp SSE về client. FastAPI không lưu lịch sử và không kiểm tra quyền sở hữu `course_id`; endpoint này dành cho backend nội bộ, không đưa trực tiếp ra client công khai.

`POST /api/v1/chat` nhận JSON, trả `text/event-stream`:

```json
{
  "message": "Cho tôi một ví dụ đơn giản",
  "history": [
    {"role": "user", "content": "HTML5 semantic tags là gì?"},
    {"role": "assistant", "content": "Đó là các thẻ thể hiện ý nghĩa nội dung như header, main, article."}
  ]
}
```

- Chat tổng quát: không gửi `course_id`; không gọi Qdrant.
- Hỏi tài liệu khóa học: gửi thêm `course_id`; retriever lọc chính xác khóa học đó. Không có dữ liệu thì chatbot được hướng dẫn nói rõ giới hạn.
- Hỏi khóa học sinh từ yêu cầu: Spring Boot gửi `course_id` và `course_materials` là JSON học liệu `CourseGenerationResponse` đã lưu. Hai course_id phải khớp. Trường hợp này dùng học liệu đó, không gọi Qdrant (vì chế độ tạo bằng requirement không lưu vectors). Nếu gửi học liệu cho khóa học tài liệu thì học liệu được ưu tiên thay cho truy xuất tài liệu gốc.
- `history` chứa tối đa 20 tin nhắn cũ theo thứ tự thời gian, không lặp lại `message` hiện tại; chỉ chấp nhận role `user` và `assistant`. Giới hạn 8000 ký tự mỗi tin, tổng history 32000, JSON course_materials 48000.

Ví dụ các frame SSE (mỗi frame kết thúc bằng dòng trống):

```text
event: start
data: {"course_id": null}

event: delta
data: {"content": "Ví dụ: "}

event: delta
data: {"content": "<main>Nội dung chính</main>"}

event: done
data: {"content": "Ví dụ: <main>Nội dung chính</main>"}
```

Spring Boot chuyển tiếp từng event ngay khi nhận, không gom toàn bộ response. Client nối các `delta.content` để hiển thị; `done.content` là bản đầy đủ, không nối thêm vào các delta. Spring Boot lưu câu trả lời hoàn chỉnh khi nhận `done`, gắn với conversation_id mà Spring Boot quản lý. API không tự phát lại yêu cầu khi kết nối đứt.

Input sai trả HTTP 422. Nếu lỗi sau khi bắt đầu stream, HTTP có thể đã là 200 nhưng stream sẽ gửi `event: error` với `code=CHAT_FAILED`, không có `done`. Spring Boot cần đánh dấu trả lời dở dang/thất bại, kể cả khi kết nối đóng mà chưa nhận done. Log chi tiết ở server; client chỉ nhận thông báo chung. Khi client hủy, Spring Boot cần hủy subscription tới FastAPI để đóng stream LLM.

Client dùng cơ chế đọc stream hỗ trợ POST JSON (ví dụ fetch); không dùng EventSource mặc định vốn gửi GET. Tắt buffering tại reverse proxy cho route chat. API tạo học liệu hiện có vẫn trả JSON một lần như trước.

## Kiểm thử

```powershell
python -m unittest discover -v
```

Bộ test tự động giả lập LLM và không gọi DeepSeek. Các script `check_qdrant_server.py` và `check_real_documents.py` dùng Qdrant thật với collection thử riêng và tự dọn collection sau kiểm tra.

### E2E với LLM thật

```powershell
python test_api.py
```

Cần Qdrant đang chạy và API key hợp lệ. Script tự mở Uvicorn trên cổng localhost trống, thử requirement HTML5 độc lập rồi upload tài liệu Zero Trust trong `papers/`, gọi LLM thật (có thể phát sinh phí), kiểm tra schema và dọn file tạm. Chế độ requirement được kiểm tra không tạo collection. Collection thử riêng của chế độ tài liệu được xóa và server thử được tắt sau khi chạy. Không cần mở server trước.

Kết quả riêng ở `test_results/requirement_html5.json` và `test_results/document_zero_trust.json` (`course_response.json` cũng giữ bản kết quả tài liệu); báo cáo HTTP, số học liệu, thời gian và cleanup ở `test_results/e2e_report.json`. Thư mục này không được đưa vào Git. E2E kiểm tra đủ thành phần, cả hai dạng câu ôn tập và schema; không chấm tự động độ chính xác toàn bộ nội dung hay ép chính xác số câu yêu cầu trong prompt.
