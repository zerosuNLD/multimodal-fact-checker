# Fact-Checker Agent

Dự án Fact-Checker (Hệ thống kiểm chứng sự thật) bao gồm một Backend FastAPI và Frontend Next.js. 

## Cấu trúc thư mục cốt lõi
- `agent/`: Các module liên quan đến việc suy luận của Agent (ReAct agent, state, graph, v.v.)
- `crawl/`: Các công cụ thu thập thông tin từ web.
- `frontend/`: Giao diện người dùng được xây dựng bằng Next.js.
- `llm/`: Cấu hình và tích hợp mô hình ngôn ngữ lớn (LLM).
- `search/`: Các công cụ tìm kiếm.
- `similarity/`: Các mô hình so sánh độ tương đồng (bao gồm LongCLIP).
- `fastapi_server.py`: File khởi chạy server Backend (FastAPI).
- `report/`: Thư mục chứa các kết quả báo cáo (đã được ignore khỏi git).
- `feedback/`: Thư mục chứa logs phản hồi (đã được ignore khỏi git).
- `temp_uploads/`: Thư mục lưu trữ các file upload tạm thời (đã được ignore khỏi git).

## Yêu cầu hệ thống
- Python 3.10+
- Node.js 18+ (Dành cho Frontend)
- Model LongCLIP (Cần được tải và đặt vào `similarity/checkpoints/longclip-B.pt`)

## Hướng dẫn cài đặt và chạy ứng dụng

### 1. Cài đặt và khởi chạy Backend (FastAPI)
Backend cung cấp API để phân tích claim (thông tin cần kiểm chứng) và hỗ trợ SSE streaming cho tiến trình phân tích.

Mở terminal tại thư mục gốc của dự án (`/factchecker`):

```bash
# 1. (Tuỳ chọn) Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # (hoặc `venv\Scripts\activate` trên Windows)

# 2. Cài đặt các thư viện cần thiết (nếu có requirements.txt thì chạy pip install -r requirements.txt)
pip install fastapi uvicorn torch python-dotenv pydantic python-multipart pyngrok

# 3. Chép cấu hình biến môi trường
# Bạn cần có file .env ở thư mục gốc chứa các API Key cần thiết (ví dụ: OPENAI_API_KEY, NGROK_AUTHTOKEN, v.v.)

# 4. Chạy server backend
python fastapi_server.py
# Server sẽ khởi chạy tại: http://0.0.0.0:8000
```

### 2. Cài đặt và khởi chạy Frontend (Next.js)
Frontend là giao diện người dùng để gửi yêu cầu và nhận báo cáo.

Mở một terminal khác và truy cập vào thư mục `frontend`:

```bash
cd frontend

# 1. Cài đặt các dependencies
npm install

# 2. Khởi chạy server phát triển
npm run dev
# Frontend sẽ khởi chạy tại: http://localhost:3000
```

## Chú ý
- Hệ thống cần tải trước model LongCLIP và đặt tại thư mục `similarity/checkpoints/longclip-B.pt` để backend có thể tải thành công.
- Các file rác hoặc kết quả chạy cục bộ (như report, feedback, ảnh test) đã được loại bỏ khi commit code lên git (được cấu hình trong `.gitignore`).
