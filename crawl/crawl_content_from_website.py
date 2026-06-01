import requests
import newspaper
from newspaper import Article
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote
from setting import scraping_dog_api_key

def scrape_article_with_fallback(target_url, dog_api_key=scraping_dog_api_key):
    print(f"[*] Bắt đầu xử lý URL: {target_url}")

    try:
        print(" -> Thử cào trực tiếp (Newspaper4k)...")
        article = newspaper.article(
            target_url
    )

        if article.text and len(article.text.strip()) > 100:
            print(" -> ✔️ Lấy dữ liệu bằng Newspaper4k.")
            return {
                "raw_html": article.html,
                "title": article.title,
                "text": article.text,
                "all_images": article.images,
                "snippet": article.meta_description,
                "method_used": "newspaper4k"
            }
        else:
            print(" ->⚠️ Nội dung quá ngắn hoặc bị chặn. Chuyển sang ScrapingDog...")
            
    except Exception as e:
        print(f" -> ⛔ Cào trực tiếp thất bại ({e}). Chuyển sang ScrapingDog...")

    # ---------------------------------------------------------
    # BƯỚC 2: Fallback dùng ScrapingDog API
    # ---------------------------------------------------------
    try:
        print(" -> Đang gọi ScrapingDog API...")
        url_api = "https://api.scrapingdog.com/scrape"

        params = {
            "api_key": dog_api_key,
            "url": target_url,
            "dynamic": "false" 
        }
        response = requests.get(url_api, params=params)
        
        if response.status_code == 200:
            raw_html = response.text
    
            
            article_dog = Article(
                target_url            
            )
            
            article_dog.download(input_html=raw_html)       

            article_dog.parse()
            
            print(" -> ✔️ Đã lấy và bóc tách dữ liệu qua ScrapingDog.")
            return {
                "raw_html": raw_html,
                "title": article_dog.title,
                "text": article_dog.text,
                "all_images": article_dog.images,
                "snippet": article_dog.meta_description,
                "method_used": "Scrapingdog"
            }
        else:
            return {"error": f"ScrapingDog cũng thất bại. Mã lỗi API: {response.status_code}"}
            
    except Exception as e:
        return {"error": f"Lỗi ở bước ScrapingDog: {e}"}


def extract_image_caption(raw_text: str, image_url: str):
    """
    Tìm caption của 1 ảnh trong HTML bài báo.

    Parameters
    ----------
    raw_text : str
        HTML thô của bài báo.
    image_url : str
        URL ảnh cần tìm caption.

    Returns
    -------
    str | None
        Caption nếu tìm thấy, ngược lại None.
    """
    if not raw_text or not image_url:
        return None

    soup = BeautifulSoup(raw_text, "html.parser")

    def normalize_url(u: str) -> str:
        if not u:
            return ""
        u = u.strip()
        u = urljoin(image_url, u)  # xử lý URL tương đối
        parsed = urlparse(u)
        # bỏ query/fragment để so khớp tốt hơn
        normalized = parsed._replace(query="", fragment="").geturl()
        return unquote(normalized)

    def clean_text(text: str) -> str:
        if not text:
            return None
        text = " ".join(text.split())
        return text if text else None

    target_url = normalize_url(image_url)
    target_name = urlparse(target_url).path.split("/")[-1]

    def matches_image(img_tag) -> bool:
        candidates = [
            img_tag.get("src"),
            img_tag.get("data-src"),
            img_tag.get("data-lazy-src"),
            img_tag.get("data-original"),
        ]

        # srcset có thể chứa nhiều URL
        srcset = img_tag.get("srcset")
        if srcset:
            for part in srcset.split(","):
                candidates.append(part.strip().split(" ")[0])

        for c in candidates:
            if not c:
                continue
            c_norm = normalize_url(c)
            if c_norm == target_url:
                return True
            if urlparse(c_norm).path.split("/")[-1] == target_name:
                return True
        return False

    def caption_from_img(img_tag):
        # Ưu tiên thuộc tính gắn trực tiếp trên ảnh
        for attr in ("alt", "title", "aria-label"):
            val = clean_text(img_tag.get(attr))
            if val:
                return val

        # Ưu tiên figcaption của figure chứa ảnh
        figure = img_tag.find_parent("figure")
        if figure:
            figcaption = figure.find("figcaption")
            if figcaption:
                val = clean_text(figcaption.get_text(" ", strip=True))
                if val:
                    return val

        # Fallback: các thẻ gần ảnh
        parent = img_tag.parent
        if parent:
            for tag in parent.find_all(["figcaption", "p", "span"], recursive=False):
                val = clean_text(tag.get_text(" ", strip=True))
                if val and len(val) <= 300:
                    return val

        for sib in [img_tag.find_next_sibling(), img_tag.find_previous_sibling()]:
            if sib and hasattr(sib, "get_text"):
                val = clean_text(sib.get_text(" ", strip=True))
                if val and len(val) <= 300:
                    return val

        return None

    # 1) Tìm ảnh khớp chính xác
    for img in soup.find_all("img"):
        if matches_image(img):
            caption = caption_from_img(img)
            if caption:
                return caption

    # 2) Fallback theo tên file ảnh nếu URL không khớp tuyệt đối
    for img in soup.find_all("img"):
        candidates = [
            img.get("src"),
            img.get("data-src"),
            img.get("data-lazy-src"),
            img.get("data-original"),
        ]
        for c in candidates:
            if not c:
                continue
            c_name = urlparse(normalize_url(c)).path.split("/")[-1]
            if c_name == target_name:
                caption = caption_from_img(img)
                if caption:
                    return caption

    return None