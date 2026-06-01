from urllib.parse import urlparse
from .exclude import EXCLUDE_DOMAINS
from setting import serp_api_key

# ── serpapi version compatibility ────────────────────────────────────
# v2+ uses SerpApiClient(params, engine=).get_dict()
# v1.x uses Client(api_key=).search(params)
try:
    from serpapi import SerpApiClient as _SerpClient
    _IS_V2 = True
except ImportError:
    from serpapi import Client as _SerpClient
    _IS_V2 = False


def _serp_search(search_params: dict, engine: str = "google_reverse_image") -> dict:
    """Execute a serpapi search, compatible with both v1.x and v2+."""
    if _IS_V2:
        client = _SerpClient(search_params, engine=engine)
        return client.get_dict()
    else:
        client = _SerpClient(api_key=search_params["api_key"])
        return client.search(search_params)


def get_links_from_image(image_url, max_results=10, api_key=serp_api_key, engine="google_reverse_image"):
    try:
        search_params = {
            "api_key": api_key,
            "engine": engine,
        }

        if engine == "google_lens":
            search_params["url"] = image_url
        else:
            search_params["image_url"] = image_url

        results = _serp_search(search_params, engine)

        # Trích xuất mảng kết quả tuỳ theo engine
        if engine == "google_lens":
            raw_matches = results.get("visual_matches", [])
        else:
            raw_matches = results.get("image_results", [])

        def is_allowed(link: str) -> bool:
            domain = urlparse(link).netloc.lower().replace("www.", "")
            return not any(excluded in domain for excluded in EXCLUDE_DOMAINS)

        matched_results = [
            {
                "title": match.get("title", "Không có tiêu đề"),
                "link": match.get("link", "")
            }
            for match in raw_matches
            if match.get("link") and is_allowed(match.get("link", ""))
        ]

        if max_results is not None:
            matched_results = matched_results[:max_results]

        return matched_results

    except Exception as e:
        print(f"Đã xảy ra lỗi khi search với {engine}: {e}")
        return 