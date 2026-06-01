import json
import concurrent.futures
import torch

from llm.prompt.create_query import generate_search_queries_prompt
from llm.call_llm import call_llm
from search.search_text import search_text
from similarity.retrieve_related_text_for_text import retrieve_related_text_for_text
from crawl.crawl_content_from_website import scrape_article_with_fallback
from similarity.chunking import chunk_article_text
import similarity.model.longclip as longclip
from llm.prompt.create_summary_prompt import generate_summary_prompt_text_only

def get_search_queries(claim: str) -> list:
    """Gọi LLM để tạo danh sách truy vấn từ claim và parse JSON."""
    system_prompt, user_prompt = generate_search_queries_prompt(claim)
    
    llm_response = call_llm(
        SYSTEM_PROMPT=system_prompt,
        USER_PROMPT=user_prompt
    )
    print("Generated Search Queries (Raw):")
    print(llm_response)

    try:
        clean_text = llm_response.strip().strip("```json").strip("```").strip()
        search_queries = json.loads(clean_text)
        return search_queries
    except Exception as e:
        print(f"Lỗi khi parse JSON: {e}")
        return []

def run_parallel_searches(search_queries: list) -> dict:
    """Thực hiện tìm kiếm Google song song cho danh sách truy vấn."""
    results = {}
    
    if not search_queries or not isinstance(search_queries, list):
        print("Không có danh sách truy vấn hợp lệ để tìm kiếm.")
        return results

    print(f"\n[*] Bắt đầu tìm kiếm {len(search_queries)} queries...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(search_queries)) as executor:
        future_to_query = {executor.submit(search_text, q): q for q in search_queries}
        
        for future in concurrent.futures.as_completed(future_to_query):
            query = future_to_query[future]
            try:
                data = future.result()
                results[query] = data
                print(f"[OK] Đã hoàn thành tìm kiếm cho: {query}")
            except Exception as exc:
                print(f"[Lỗi] Query '{query}' sinh ra ngoại lệ: {exc}")
                results[query] = None
                
    return results

def display_results(results: dict):
    """In kết quả tìm kiếm ra màn hình."""
    if not results:
        return

    print("\n" + "="*40)
    for q, res in results.items():
        print(f"\nQuery: {q}")
        print(f"Số lượng kết quả: {len(res) if res else 0}") 
        if res:
            for item in res:
                print(f" -> Link: {item.get('link')}")

def _process_single_url_full_parallel(query, url, snippet, model, device, k_chunks):
    """
    Hàm worker: Cào và trích xuất thông tin, KHÔNG dùng khóa bảo vệ GPU.
    Chạy song song hoàn toàn.
    """
    try:
        # 1. Cào nội dung website
        scraped_data = scrape_article_with_fallback(url)
        
        if "text" in scraped_data and scraped_data["text"]:
            article_text = scraped_data["text"]
            
            # 2. Băm nhỏ văn bản
            chunks = chunk_article_text(article_text, min_words=40)
            
            # 3. Trích xuất K câu liên quan nhất bằng LongCLIP
            if chunks:
                related_chunks = retrieve_related_text_for_text(
                    query_text=query, 
                    chunks=chunks, 
                    model=model, 
                    device=device, 
                    k=k_chunks
                )
                
                SYSTEM_PROMPT, USER_PROMPT = generate_summary_prompt_text_only(related_chunks)
                summary = call_llm(SYSTEM_PROMPT=SYSTEM_PROMPT, USER_PROMPT=USER_PROMPT)
                
                # Trả về chỉ 4 trường cần thiết cho agent
                result_dict = {
                    "title": scraped_data.get("title", ""),
                    "snippet": snippet or scraped_data.get("snippet", ""),
                    "summary": summary,
                    "url": url,
                }
                
                return query, url, result_dict, None
            else:
                return query, url, None, "Bài viết sau khi chunking bị rỗng."
        else:
            return query, url, None, "Không cào được text."
    except Exception as e:
        return query, url, None, f"Lỗi: {str(e)}"


def extract_information_from_links(search_results: dict, model, device, k_chunks=15, max_links_per_query=5):
    """
    Duyệt qua các link, cào bài báo, băm nhỏ và trích xuất câu liên quan SONG SONG TOÀN BỘ.
    Trả về dữ liệu có kèm theo URL nguồn.
    """
    extracted_data = {query: [] for query in search_results.keys()}
    
    # Gom tất cả các URL cần xử lý thành danh sách tasks
    tasks = []
    for query, links_data in search_results.items():
        if not links_data:
            continue
        
        # Lấy top các link đầu tiên để cào
        top_links = links_data[:max_links_per_query]
        for item in top_links:
            url = item.get("link")
            snippet = item.get("snippet", "")
            if url:
                tasks.append((query, url, snippet))

    if not tasks:
        print("Không có URL nào hợp lệ để cào dữ liệu.")
        return extracted_data

    num_workers = min(len(tasks), 5) 
    print(f"\n[*] Bắt đầu cào và trích xuất SONG SONG HOÀN TOÀN cho {len(tasks)} URLs ({num_workers} luồng)...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {
            executor.submit(_process_single_url_full_parallel, q, u, s, model, device, k_chunks): (q, u)
            for q, u, s in tasks
        }
        
        for future in concurrent.futures.as_completed(future_to_task):
            query_str, url_str = future_to_task[future]
            try:
                q, u, result_dict, error_msg = future.result()
                if error_msg:
                    print(f" -> ⚠️ Bỏ qua [{u}]: {error_msg}")
                elif result_dict:
                    extracted_data[q].append(result_dict)
                    print(f" -> ✔️ Xong: {u} (Trích xuất {len(result_dict.get('chunks', []))} chunks)")
            except Exception as exc:
                print(f" -> ⛔ Thread crash tại [{url_str}]: {exc}")
                
    return extracted_data

def pipepline_retrieve_related_inf_for_claim(
    device,
    model,
    preprocess,
    claim=""
):
    
    # Bước 1: Lấy danh sách câu hỏi
    queries = get_search_queries(claim)
    
    # Bước 2: Tìm kiếm song song
    search_results = run_parallel_searches(queries)
    
    # Bước 3: Hiển thị danh sách link tìm được
    display_results(search_results)


    # Bước 5: Cào song song và trích xuất thông tin
    final_data = extract_information_from_links(
        search_results=search_results, 
        model=model, 
        device=device, 
        k_chunks=15,             
        max_links_per_query=5   
    )
    
    return final_data