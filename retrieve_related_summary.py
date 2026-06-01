import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm.call_llm import call_llm
from llm.prompt.find_related_summary_prompt import generate_find_related_summary_prompt

def filter_summaries_for_single_image(image_group: list) -> list:
    """
    Hàm xử lý cho 1 ảnh duy nhất.
    Tự tìm summary có điểm cao nhất của RIÊNG ảnh này để làm mốc lọc.
    """
    image_path = image_group[0]
    summaries = image_group[1]
    
    # Nếu ảnh này không có summary nào từ đầu, trả về rỗng luôn
    if not summaries:
        return [image_path, []]
        
    # 1. TÌM SUMMARY CÓ ĐIỂM CAO NHẤT CỦA RIÊNG ẢNH NÀY
    highest_summary_for_this_image = max(summaries, key=lambda x: x.get("max_score", 0), default=None)
    
    if not highest_summary_for_this_image:
        return [image_path, []]

    print(f"[*] Ảnh '{image_path}': Dùng summary cao nhất (Score: {highest_summary_for_this_image.get('max_score', 0):.4f}) để làm mốc lọc.")
        
    # 2. Tạo prompt so sánh mốc này với các summary còn lại của ảnh
    system_p, user_p = generate_find_related_summary_prompt(highest_summary_for_this_image, summaries)
    
    response_text = call_llm(system_p, user_p, temperature=0)
    
    related_summaries = []
    try:
        # Làm sạch chuỗi trả về để chống lỗi parse JSON
        match = re.search(r'\[.*\]', response_text.strip(), re.DOTALL)
        if match:
            json_str = match.group(0)
            related_indexes = json.loads(json_str)
            
            # Lấy ra các summary có index nằm trong mảng LLM trả về
            for idx in related_indexes:
                if 0 <= idx < len(summaries):
                    related_summaries.append(summaries[idx])
        else:
            print(f"⚠️ LLM không trả về định dạng mảng cho ảnh {image_path}. Output: {response_text}")

    except json.JSONDecodeError as e:
        print(f"❌ Lỗi parse JSON từ LLM cho ảnh {image_path}: {e}\nOutput: {response_text}")
    except Exception as e:
        print(f"❌ Lỗi không xác định khi xử lý ảnh {image_path}: {e}")
        
    return [image_path, related_summaries]

def get_related_summaries_from_all_results(ket_qua_tong: list, max_workers: int = 5) -> dict:
    """
    Hàm chính: Chạy song song bộ lọc cho toàn bộ danh sách ảnh.
    Output là dạng Dict: {"image_path": [các_summary_đã_lọc]}
    """
    print(f"\n[*] Đang tiến hành lọc summary liên quan độc lập cho từng ảnh ...")
    final_filtered_results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit task (Không cần truyền highest_summary vào nữa)
        futures = [
            executor.submit(filter_summaries_for_single_image, group)
            for group in ket_qua_tong
        ]
        
        for future in futures:
            try:
                result_group = future.result()
                image_path = result_group[0]
                filtered_summaries = result_group[1]
                
                # Chỉ đưa vào kết quả nếu ảnh này CÒN ÍT NHẤT 1 summary liên quan
                if len(filtered_summaries) > 0:
                    final_filtered_results[image_path] = filtered_summaries
                    
            except Exception as e:
                print(f"❌ Lỗi trong luồng xử lý chính: {e}")
                
    print(f"✔️ Đã lọc xong! Giữ lại {len(final_filtered_results)} ảnh có chứa thông tin liên quan.")
    return final_filtered_results

def get_highest_score_summary(ket_qua_tong: list) -> dict:
    """
    Duyệt toàn bộ kết quả để tìm summary có max_score cao nhất.
    Trả về dict chứa thông tin summary đó và đường dẫn ảnh gốc.
    """
    highest_summary = None
    max_found_score = -1.0

    for image_group in ket_qua_tong:
        image_path = image_group[0]
        summaries = image_group[1]

        for s in summaries:
            current_score = s.get("max_score", 0)
            
            if current_score > max_found_score:
                max_found_score = current_score
                highest_summary = s.copy()
                highest_summary["source_image"] = image_path 

    if highest_summary:
        print(f"✔️ Đã tìm thấy summary có score cao nhất: {max_found_score:.4f}")
    else:
        print("⚠️ Không tìm thấy summary nào trong dữ liệu.")
        
    return highest_summary