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

### 1. Chuẩn bị (Yêu cầu bắt buộc)
1. **Model LongCLIP:** Hệ thống cần mô hình LongCLIP. Hãy tải checkpoint của LongCLIP và đặt file vào thư mục: `similarity/checkpoints/longclip-B.pt`
2. **File `.env`:** Tạo file `.env` ở thư mục gốc (nếu chưa có) và điền các API key cần thiết (ví dụ: `OPENAI_API_KEY`, v.v.).

### 2. Chạy ứng dụng bằng Docker (Khuyên dùng)
Cách đơn giản nhất để khởi chạy toàn bộ hệ thống là sử dụng Docker Compose. Đảm bảo bạn đã cài đặt Docker và Docker Compose trên máy tính.

Mở terminal tại thư mục gốc của dự án (`/factchecker`) và chạy lệnh:

```bash
docker-compose up --build -d
```

Lệnh này sẽ tự động:
- Xây dựng và khởi chạy Backend (FastAPI) tại **http://localhost:8000**
- Xây dựng và khởi chạy Frontend (Next.js) tại **http://localhost:3000**

Để dừng ứng dụng, chạy lệnh:
```bash
docker-compose down
```

---

### 3. Cài đặt và khởi chạy thủ công (Nếu không dùng Docker)

#### Khởi chạy Backend (FastAPI)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python fastapi_server.py
```

#### Khởi chạy Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev
```

## Chú ý
- Hệ thống cần tải trước model LongCLIP và đặt tại thư mục `similarity/checkpoints/longclip-B.pt` để backend có thể tải thành công.
- Các file rác hoặc kết quả chạy cục bộ (như report, feedback, ảnh test) đã được loại bỏ khi commit code lên git (được cấu hình trong `.gitignore`).
