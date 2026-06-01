"""Tool registry cho ReAct agent.

2 tools:
  - search_text(query)  → [{title, snippet, summary, url}]
  - search_image(image_path) → [{title, snippet, summary, url}]
"""

import json
from typing import Any

from setting import scraping_dog_api_key, serp_api_key
from search.search_text import search_text as _google_search
from crawl.crawl_content_from_website import scrape_article_with_fallback
from similarity.chunking import chunk_article_text
from similarity.retrieve_related_text_for_text import retrieve_related_text_for_text
from llm.prompt.create_summary_prompt import generate_summary_prompt_text_only
from llm.call_llm import call_llm


# ── Shared model refs (set từ main) ──────────────────────────────────
_model      = None
_preprocess = None
_device     = "cpu"


def set_models(model, preprocess, device: str):
    global _model, _preprocess, _device
    _model      = model
    _preprocess = preprocess
    _device     = device


# ═══════════════════════════════════════════════════════════════════════
# Tool: search_text
# ═══════════════════════════════════════════════════════════════════════
def search_text(query: str, max_urls: int = 4) -> list[dict]:
    """Tìm kiếm Google với 1 câu truy vấn, cào bài báo, trích xuất nội dung.

    Args:
        query: Câu truy vấn Google do agent tự đặt.
        max_urls: Số URL tối đa để cào (mặc định 4).

    Returns:
        list[dict] với mỗi phần tử: {title, snippet, summary, url}
    """
    results = []
    try:
        search_hits = _google_search(query, num_results=max_urls)
    except Exception as e:
        return [{"error": f"Lỗi search: {e}"}]

    for item in search_hits[:max_urls]:
        url     = item.get("link", "")
        title   = item.get("title", "")
        snippet = item.get("snippet", "")
        if not url:
            continue
        try:
            scraped = scrape_article_with_fallback(url)
            text = scraped.get("text", "")
            if not text:
                continue
            chunks = chunk_article_text(text, min_words=40)
            if not chunks:
                continue
            related = retrieve_related_text_for_text(
                query_text=query,
                chunks=chunks,
                model=_model,
                device=_device,
                k=10,
            )
            if not related:
                continue
            sys_p, usr_p = generate_summary_prompt_text_only(related)
            summary = call_llm(sys_p, usr_p)
            results.append({
                "title":   title or scraped.get("title", ""),
                "snippet": snippet,
                "summary": summary,
                "url":     url,
            })
        except Exception:
            continue

    return results


# ═══════════════════════════════════════════════════════════════════════
# Tool: search_image
# ═══════════════════════════════════════════════════════════════════════
def search_image(image_path: str, max_urls: int = 10) -> list[dict]:
    """Reverse image search + cào bài báo chứa ảnh, trích xuất nội dung.

    Args:
        image_path: Đường dẫn file ảnh cục bộ.
        max_urls: Số URL tối đa (mặc định 10).

    Returns:
        list[dict] với mỗi phần tử: {title, snippet, summary, url}
    """
    from pipeline_extracted__related_inf_for_img import (
        pipeline_extracted_related_information_for_images,
    )
    from retrieve_related_summary import get_related_summaries_from_all_results

    try:
        raw = pipeline_extracted_related_information_for_images(
            images=[image_path],
            dog_api_key=scraping_dog_api_key,
            serp_api_key=serp_api_key,
            model=_model,
            preprocess=_preprocess,
            device=_device,
            max_urls_per_image=max_urls,
            max_concurrent_images=1,
        )
        filtered = get_related_summaries_from_all_results(raw, max_workers=3)
    except Exception as e:
        return [{"error": f"Lỗi image search: {e}"}]

    results = []
    for _img, summaries in filtered.items():
        for s in summaries:
            if s.get("url") and s.get("summary"):
                results.append({
                    "title":   s.get("title", ""),
                    "snippet": s.get("snippet", ""),
                    "summary": s.get("summary", ""),
                    "url":     s.get("url", ""),
                })
    return results


# ═══════════════════════════════════════════════════════════════════════
# Tool: extract_text_from_image
# ═══════════════════════════════════════════════════════════════════════
_EASYOCR_READER = None
_VIETOCR_PREDICTOR = None

def extract_text_from_image(image_path: str) -> dict:
    """Sử dụng EasyOCR (để dò vùng chữ) và VietOCR (để đọc chữ) trong hình ảnh.
    
    Args:
        image_path: Đường dẫn đến file ảnh cục bộ.
        
    Returns:
        dict chứa văn bản chữ viết được trích xuất.
    """
    global _EASYOCR_READER, _VIETOCR_PREDICTOR
    import os
    
    if not os.path.exists(image_path):
        return {"error": f"Không tìm thấy file ảnh tại đường dẫn: {image_path}"}
        
    try:
        import cv2
        import easyocr
        from PIL import Image
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg
        
        if _EASYOCR_READER is None:
            print("[OCR] Loading EasyOCR detector...")
            _EASYOCR_READER = easyocr.Reader(['vi'], gpu=False)
            
        if _VIETOCR_PREDICTOR is None:
            print("[OCR] Loading VietOCR model...")
            config = Cfg.load_config_from_name('vgg_transformer')
            config['device'] = 'cpu'
            _VIETOCR_PREDICTOR = Predictor(config)
            
        result = _EASYOCR_READER.detect(image_path)
        horizontal_list = result[0][0]
        
        if not horizontal_list:
            return {"extracted_text": ""}
            
        img = cv2.imread(image_path)
        if img is None:
            return {"error": f"Không thể đọc file ảnh: {image_path}"}
        
        # Convert grayscale (if needed) or BGR to RGB safely
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        horizontal_list.sort(key=lambda box: box[2])
        extracted_text_chunks = []
        
        for box in horizontal_list:
            x_min, x_max, y_min, y_max = [int(v) for v in box]
            
            h, w, _ = img.shape
            pad = 2
            y_min = max(0, y_min - pad)
            y_max = min(h, y_max + pad)
            x_min = max(0, x_min - pad)
            x_max = min(w, x_max + pad)
            
            cropped = img[y_min:y_max, x_min:x_max]
            if cropped.size == 0:
                continue
                
            pil_img = Image.fromarray(cropped)
            text = _VIETOCR_PREDICTOR.predict(pil_img)
            if text.strip():
                extracted_text_chunks.append(text)
                
        extracted_text = " ".join(extracted_text_chunks)
        return {"extracted_text": extracted_text}
        
    except Exception as e:
        return {"error": f"Lỗi extract_text_from_image (VietOCR): {e}"}

# ═══════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════
TOOL_REGISTRY: dict[str, Any] = {
    "search_text":  search_text,
    "search_image": search_image,
    "extract_text_from_image": extract_text_from_image,
}


def execute_tool(name: str, args: dict) -> str:
    """Thực thi tool theo tên và trả về kết quả dạng JSON string."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return json.dumps({"error": f"Tool '{name}' không tồn tại."}, ensure_ascii=False)
    try:
        result = fn(**args)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
