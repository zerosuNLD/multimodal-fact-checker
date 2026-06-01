import torch
from .model import longclip

# LongCLIP hard limit: 248 tokens
def _truncate(texts: list[str]) -> list[str]:
    """Cắt chunk về ~180 từ (≈ 230 token) trước khi tokenize."""
    out = []
    for t in texts:
        words = t.split()
        if len(words) > 180:
            t = " ".join(words[:180])
        out.append(t)
    return out

def retrieve_related_text_for_text(query_text, chunks, model, device, k=10, threshold=0.2):
    """
    Trích xuất các đoạn văn bản liên quan đến một đoạn văn bản truy vấn (query text).
    - k: Số lượng chunk tối đa cần lấy.
    - threshold: Ngưỡng độ tương đồng tối thiểu. Nếu max similarity của toàn bộ chunks 
                 mà thấp hơn ngưỡng này, sẽ bỏ qua toàn bộ (trả về mảng rỗng).
    """
    if not chunks:
        return []
    
    k = min(k, len(chunks))
    safe_chunks = _truncate(chunks)
    safe_query  = _truncate([query_text])
    text_tokens  = longclip.tokenize(safe_chunks).to(device)
    query_tokens = longclip.tokenize(safe_query).to(device)
    
    with torch.no_grad():
        if device == "cuda":
            stream_txt = torch.cuda.Stream()
            stream_query = torch.cuda.Stream()

            with torch.cuda.stream(stream_txt):
                text_features = model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
            with torch.cuda.stream(stream_query):
                query_features = model.encode_text(query_tokens)
                query_features /= query_features.norm(dim=-1, keepdim=True)

            torch.cuda.synchronize()
        else:
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            query_features = model.encode_text(query_tokens)
            query_features /= query_features.norm(dim=-1, keepdim=True)
        
    # Tính cosine similarity giữa query và từng chunk
    similarities = (query_features @ text_features.T).squeeze(0)
    
    # 1. KIỂM TRA NGƯỠNG MAX SIMILARITY ĐỂ BỎ QUA LINK
    max_sim = torch.max(similarities).item()
    if max_sim < threshold:
        # print(f"[-] Bỏ qua do max similarity ({max_sim:.4f}) thấp hơn threshold ({threshold})")
        return []
    
    # 2. Lọc và lấy top K
    # Sử dụng torch.argsort để sắp xếp trực tiếp trên Tensor
    ranked_indices = torch.argsort(similarities, descending=True)[:k]
    
    related_chunks = []
    for idx_tensor in ranked_indices:
        idx = idx_tensor.item()
        score = similarities[idx].item()
        
        # Chỉ lấy những chunk thực sự vượt qua ngưỡng
        if score >= threshold:
            related_chunks.append(chunks[idx])
            
    return related_chunks