import os
import requests
import base64
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import shutil

def download_images_from_urls(url_list, folder_path="image", max_workers=20):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }

    def download_single_image(url):
        try:
            path = urlparse(url).path
            ext = os.path.splitext(path)[1]
            if not ext:
                ext = ".jpg" 

            url_encoded = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
            
            file_name = f"{url_encoded[:200]}{ext}"
            save_to = os.path.join(folder_path, file_name)

            response = requests.get(url, headers=headers, stream=True, timeout=15)
            
            if response.status_code == 200:
                with open(save_to, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                return f"SUCCESS: {file_name}"
            return f"FAILED: {url}"
                
        except Exception as e:
            return f"ERROR: {url} ({str(e)})"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(download_single_image, url_list))
    
    return results

def get_url_from_filename(filename):
    """Giải mã tên file để lấy lại URL gốc"""
    name_without_ext = os.path.splitext(filename)[0]
    
    padding = len(name_without_ext) % 4
    if padding:
        name_without_ext += "=" * (4 - padding)
        
    url_bytes = base64.urlsafe_b64decode(name_without_ext.encode())
    return url_bytes.decode()



def clear_image_folder(folder_path):
    if os.path.exists(folder_path):
        print(f"--- Đang dọn dẹp thư mục: {folder_path} ---")
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        print(f"--- Đã dọn dẹp sạch thư mục: {folder_path} ---")
    else:
        os.makedirs(folder_path)