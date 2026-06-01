import torch
from PIL import Image
from .model import longclip

# LongCLIP hard limit: 248 tokens
_MAX_TOKENS = 248

def _truncate(texts: list[str]) -> list[str]:
    """Bảo đảm mỗi chuỗi không quá _MAX_TOKENS word-piece token.
    Truncate bằng cách cắt đười của không gian (xấp xỉ, an toàn).
    """
    out = []
    for t in texts:
        # Một từ ≈ 1.3 token trung bình; cắt ở ~180 từ để an toàn
        words = t.split()
        if len(words) > 180:
            t = " ".join(words[:180])
        out.append(t)
    return out

def retrieve_related_text_for_image(
    chunks, 
    caption, 
    image_input, 
    model,       
    preprocess,  
    device,      
    k=10
):
    """
    Trích xuất các đoạn văn bản liên quan đến hình ảnh và chú thích.
    
    Args:
        chunks (list[str]): Danh sách các đoạn văn bản (sentences/paragraphs) được tách ra từ bài báo. 
                           Đây là tập hợp các ứng viên để mô hình so khớp.
        caption (str): Nội dung chú thích (caption) đi kèm với hình ảnh. 
                       Đóng vai trò là "truy vấn văn bản" để tìm sự đồng thuận với nội dung bài báo.
        image_input (str hoặc PIL.Image): Đường dẫn đến file ảnh (string) hoặc đối tượng ảnh đã mở bằng PIL.
                                         Đóng vai trò là "truy vấn hình ảnh" để so khớp thị giác.
        model (nn.Module): Đối tượng mô hình LongCLIP đã được load sẵn (thường qua longclip.load).
                          Việc truyền model từ ngoài vào giúp tránh lãng phí RAM/VRAM khi gọi hàm nhiều lần.
        preprocess (callable): Hàm tiền xử lý hình ảnh đi kèm với mô hình CLIP. 
                              Dùng để resize, normalize ảnh về đúng định dạng đầu vào của Encoder.
        device (str): Thiết bị tính toán, thường là "cuda" (GPU) hoặc "cpu". 
                     Quyết định việc đẩy Tensor lên đâu để tính toán.
        k (int, optional): Số lượng đoạn văn bản liên quan nhất muốn trả về. Mặc định là 10.

    Returns:
        list[str]: Danh sách chứa tối đa k đoạn văn bản có độ liên quan cao nhất, 
                  đã được sắp xếp theo thứ tự ưu tiên giảm dần.
                  
    Cơ chế:
    1. Image --- CLIP ---> Vector (Image)
    2. Caption --- CLIP ---> Vector (Caption)
    3. Text Chunks --- CLIP ---> Vector (Cho từng đoạn)
    4. Tính Rank của Each Chunk theo Image và Caption
    5. Hợp nhất bằng Reciprocal Rank Fusion (RRF):
       Score = 1/(k_rrf + rank_image) + 1/(k_rrf + rank_caption)
    """
    
    # Kiểm tra nếu không có chunks nào thì trả về danh sách rỗng
    if not chunks:
        return []

    k = min(k, len(chunks))

    # Xử lý hình ảnh an toàn (thêm convert RGB chống lỗi hệ màu)
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    else:
        image = image_input

    # Truncate chunks và caption về giới hạn 248 token của LongCLIP
    safe_chunks  = _truncate(chunks)
    safe_caption = _truncate([caption])

    image_tensor   = preprocess(image).unsqueeze(0).to(device)
    text_tokens    = longclip.tokenize(safe_chunks).to(device)
    caption_tokens = longclip.tokenize(safe_caption).to(device)

    with torch.no_grad():
        if device == "cuda":
            stream_img = torch.cuda.Stream()
            stream_txt = torch.cuda.Stream()
            stream_cap = torch.cuda.Stream()

            with torch.cuda.stream(stream_img):
                image_features = model.encode_image(image_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
            with torch.cuda.stream(stream_txt):
                text_features = model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
            with torch.cuda.stream(stream_cap):
                caption_features = model.encode_text(caption_tokens)
                caption_features /= caption_features.norm(dim=-1, keepdim=True)

            torch.cuda.synchronize()
        else:
            image_features = model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            caption_features = model.encode_text(caption_tokens)
            caption_features /= caption_features.norm(dim=-1, keepdim=True)


        sim_image = (image_features @ text_features.T).squeeze(0)  
        sim_caption = (caption_features @ text_features.T).squeeze(0) 

        # Tính Rank 
        ranks_image = torch.empty_like(sim_image, dtype=torch.long)
        ranks_image[torch.argsort(sim_image, descending=True)] = torch.arange(1, len(sim_image) + 1, device=device)

        ranks_caption = torch.empty_like(sim_caption, dtype=torch.long)
        ranks_caption[torch.argsort(sim_caption, descending=True)] = torch.arange(1, len(sim_caption) + 1, device=device)


        rrf_k = 60
        rrf_scores = (1.0 / (rrf_k + ranks_image.float())) + (1.0 / (rrf_k + ranks_caption.float()))
        
        # Lấy ra Top K vị trí
        final_sorted_indices = torch.argsort(rrf_scores, descending=True)
        top_k_indices = final_sorted_indices[:k].tolist()

    return [chunks[idx] for idx in top_k_indices]