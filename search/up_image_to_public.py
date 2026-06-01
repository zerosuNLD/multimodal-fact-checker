import cloudinary
import cloudinary.uploader
from setting import cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret

def upload_image_get_url(image_path):
    cloudinary.config(
        cloud_name=cloudinary_cloud_name,
        api_key=cloudinary_api_key,
        api_secret=cloudinary_api_secret
    )

    result = cloudinary.uploader.upload(image_path)
    return result["secure_url"]


