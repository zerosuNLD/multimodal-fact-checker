import re

def chunk_article_text(article_text, min_words=40):
    """
    Tách nội dung bài báo thành các chunks dựa trên đoạn văn (xuống dòng \n).
    - article_text: (String) Toàn bộ nội dung bài báo.
    - min_words: (Int) Số từ tối thiểu cho một chunk. Nếu một đoạn văn quá ngắn 
                 (ít hơn min_words), nó sẽ tự động gộp với đoạn tiếp theo.
    """

    raw_paragraphs = re.split(r'\n+', article_text)
    
    paragraphs = [p.strip() for p in raw_paragraphs if len(p.strip()) > 0]
    
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        if not current_chunk:
            current_chunk = p
        else:
            word_count = len(current_chunk.split())
            
            if word_count < min_words:
                current_chunk += " " + p
            else:
                chunks.append(current_chunk)
                current_chunk = p
                
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks
