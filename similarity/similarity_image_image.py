import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def _process_single_image(img_input, preprocess):
    """
    Hàm phụ trợ: Mở ảnh và tiền xử lý. 
    Thêm .convert('RGB') để tránh lỗi crash khi gặp ảnh PNG có nền trong suốt (4 channels).
    """
    if isinstance(img_input, str):
        img = Image.open(img_input).convert('RGB')
    else:
        img = img_input
    return preprocess(img).unsqueeze(0)


def compare_image_similarity(image_input1, image_input2, model, preprocess, device):
    """
    So sánh độ tương đồng giữa 2 hình ảnh (Đã tối ưu song song).
    
    Args:
        image_input1: Đường dẫn hình ảnh (str) hoặc PIL Image object
        image_input2: Đường dẫn hình ảnh (str) hoặc PIL Image object
        model, preprocess, device: Truyền vào từ ngoài để tránh load lại mô hình
    
    Returns:
        float: Điểm tương đồng từ 0 đến 1 (1 = giống hệt nhau)
    """
    # 1. Song song trên CPU (I/O): Đọc 2 file ảnh cùng lúc
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(_process_single_image, image_input1, preprocess)
        future2 = executor.submit(_process_single_image, image_input2, preprocess)
        
        image_tensor1 = future1.result().to(device)
        image_tensor2 = future2.result().to(device)

    with torch.no_grad():
        if device == "cuda":
            stream1 = torch.cuda.Stream()
            stream2 = torch.cuda.Stream()
            
            with torch.cuda.stream(stream1):
                image_features1 = model.encode_image(image_tensor1)
                image_features1 /= image_features1.norm(dim=-1, keepdim=True)
            
            with torch.cuda.stream(stream2):
                image_features2 = model.encode_image(image_tensor2)
                image_features2 /= image_features2.norm(dim=-1, keepdim=True)
            
            torch.cuda.synchronize()
        else:
            image_features1 = model.encode_image(image_tensor1)
            image_features1 /= image_features1.norm(dim=-1, keepdim=True)
            
            image_features2 = model.encode_image(image_tensor2)
            image_features2 /= image_features2.norm(dim=-1, keepdim=True)
        
        # 3. Tính độ tương đồng
        similarity = (image_features1 @ image_features2.T).item()
    
    return similarity


def compare_image_similarity_batch(image_list, reference_image, model, preprocess, device):
    """
    So sánh độ tương đồng giữa 1 ảnh tham chiếu và danh sách nhiều ảnh (Đã tối ưu song song).
    """
    if not image_list:
        return []

    with ThreadPoolExecutor() as executor:
        ref_future = executor.submit(_process_single_image, reference_image, preprocess)
        list_futures = [executor.submit(_process_single_image, img, preprocess) for img in image_list]
        
        ref_tensor = ref_future.result().to(device)
        image_tensors = [f.result() for f in list_futures]
    
    image_batch = torch.cat(image_tensors, dim=0).to(device)

    with torch.no_grad():
        if device == "cuda":
            stream_ref = torch.cuda.Stream()
            stream_batch = torch.cuda.Stream()

            with torch.cuda.stream(stream_ref):
                ref_features = model.encode_image(ref_tensor)
                ref_features /= ref_features.norm(dim=-1, keepdim=True)
            
            with torch.cuda.stream(stream_batch):
                image_features = model.encode_image(image_batch)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            torch.cuda.synchronize()
        else:
            ref_features = model.encode_image(ref_tensor)
            ref_features /= ref_features.norm(dim=-1, keepdim=True)
            
            image_features = model.encode_image(image_batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        similarities = (ref_features @ image_features.T).squeeze(0)
    
    return similarities.cpu().tolist()