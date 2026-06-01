from similarity.chunking import chunk_article_text
from similarity.retrieve_related_text_for_image import retrieve_related_text_for_image
from crawl.crawl_content_from_website import scrape_article_with_fallback, extract_image_caption
from similarity.similarity_image_image import compare_image_similarity_batch
import similarity.model.longclip as longclip
from crawl.download_image import download_images_from_urls, get_url_from_filename
from search.search_image import get_links_from_image
import torch
import glob
import os
import uuid
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from search.up_image_to_public import upload_image_get_url

def process_single_url(url_can_cao, query_image, dog_api_key, model, preprocess, device):
    """Xử lý toàn bộ luồng cào bài, tải ảnh, tìm caption và trích xuất text cho 1 URL duy nhất."""
    try:
        # Cào bài báo
        ket_qua = scrape_article_with_fallback(url_can_cao, dog_api_key=dog_api_key)
        
        if 'text' not in ket_qua or not ket_qua['text']:
            return None # Bỏ qua nếu không có nội dung
            
        # Tạo thư mục tạm thời ĐỘC LẬP cho Thread này để tránh xóa nhầm file của Thread khác
        temp_folder = f"temp_image_{uuid.uuid4().hex[:8]}"
        os.makedirs(temp_folder, exist_ok=True)
        
        caption_text = ""
        
        try:
            # Tải ảnh vào thư mục tạm
            if 'all_images' in ket_qua and ket_qua['all_images']:
                download_images_from_urls(ket_qua['all_images'], folder_path=temp_folder, max_workers=5)
            
            # Quét ảnh đã tải
            image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
            downloaded_image_paths = []
            for ext in image_extensions:
                downloaded_image_paths.extend(glob.glob(os.path.join(temp_folder, ext)))
                
            # Nếu tải được ảnh, tìm ảnh có Score cao nhất để lấy Caption
            if downloaded_image_paths:
                comparison_results = compare_image_similarity_batch(
                    image_list=downloaded_image_paths,
                    reference_image=query_image,
                    model=model,
                    preprocess=preprocess,
                    device=device
                )
                
                if comparison_results:
                    max_score = max(comparison_results)
                    if max_score > 0.9: # Ngưỡng an toàn
                        best_match_idx = comparison_results.index(max_score)
                        best_image_path = downloaded_image_paths[best_match_idx]
                        
                        best_image_url = get_url_from_filename(os.path.basename(best_image_path))
                        caption_text = extract_image_caption(ket_qua.get('raw_html', ''), best_image_url)
        finally:
            # Xóa thư mục tạm ngay cả khi bị lỗi để giải phóng bộ nhớ
            shutil.rmtree(temp_folder, ignore_errors=True)
        
        # Băm nhỏ bài báo và lấy K câu liên quan nhất
        chunks_list = chunk_article_text(ket_qua['text'], min_words=40)
        
        top_related_sentences = retrieve_related_text_for_image(
            chunks=chunks_list,
            caption=caption_text,
            image_input=query_image,
            model=model,
            preprocess=preprocess,
            device=device,
            k=10
        )
        
        # Trả về chỉ 4 trường cần thiết cho agent
        combined_text = "\n".join(top_related_sentences)
        if combined_text.strip():
            # snippet: lấy từ bài báo nếu có, fallback về 2 câu đầu của summary
            raw_snippet = ket_qua.get("snippet", "")
            if not raw_snippet:
                sentences = [s.strip() for s in combined_text.split("\n") if s.strip()]
                raw_snippet = " ".join(sentences[:2])
            result_dict = {
                "title": ket_qua.get("title", ""),
                "snippet": raw_snippet,
                "summary": combined_text,
                "url": url_can_cao,
            }
            return result_dict
        return None

    except Exception as e:
        print(f"[!] Lỗi khi xử lý URL {url_can_cao}: {e}")
        return None


def process_single_query_image(query_image, dog_api_key, serp_api_key, max_urls, model, preprocess, device):
    """Tìm kiếm link và điều phối xử lý song song các URL của một bức ảnh."""
    print(f"\n[*] Đang tìm kiếm bài viết cho ảnh: {query_image}")
    test_image_url = upload_image_get_url(query_image)
    search_results = get_links_from_image(test_image_url, max_results=max_urls, api_key=serp_api_key)
    
    if not search_results:
        print(f"⚠️ Không tìm thấy bài báo nào chứa ảnh {query_image}")
        return [query_image, []]
        
    urls = [item['link'] for item in search_results]
    related_texts_for_this_image = []
    
    # Xử lý song song các URL tìm được cho ảnh này
    print(f"[*] Đang cào song song {len(urls)} URL cho ảnh {query_image}...")
    with ThreadPoolExecutor(max_workers=max_urls) as executor:
        # Submit các task
        futures = {
            executor.submit(process_single_url, url, query_image, dog_api_key, model, preprocess, device): url 
            for url in urls
        }
        
        # Thu thập kết quả khi các luồng chạy xong
        for future in as_completed(futures):
            result_text = future.result()
            if result_text:
                related_texts_for_this_image.append(result_text)
                
    return [query_image, related_texts_for_this_image]


def pipeline_extracted_related_information_for_images(
    images: list[str],
    dog_api_key: str,
    serp_api_key: str,
    model=None,
    preprocess=None,
    device: str = "cpu",
    max_urls_per_image: int = 3,
    max_concurrent_images: int = 2, # Giới hạn số lượng ảnh chạy cùng lúc để tránh sập GPU VRAM
):
    """Pipeline tổng xử lý song song toàn diện."""
    if model is None or preprocess is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[*] Đang khởi tạo mô hình LongCLIP (Chỉ 1 lần)...")
        model, preprocess = longclip.load("./similarity/checkpoints/longclip-B.pt", device=device)
    
    final_results = []
    
    print(f"\n{'='*60}\nBẮT ĐẦU XỬ LÝ SONG SONG {len(images)} ẢNH CHÍNH\n{'='*60}")
    
    with ThreadPoolExecutor(max_workers=max_concurrent_images) as executor:
        futures = [
            executor.submit(
                process_single_query_image, 
                img, dog_api_key, serp_api_key, max_urls_per_image, model, preprocess, device
            ) 
            for img in images
        ]
        
        for future in as_completed(futures):
            final_results.append(future.result())
            
    print("\n Hoàn tất trích xuất thông tin liên quan từ các ảnh!")
    return final_results
