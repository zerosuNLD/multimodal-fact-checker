from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY")

openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY")
    
serper_api_key: str = os.getenv("SERPER_API_KEY")
    
scraping_dog_api_key: str = os.getenv("SCRAPING_DOG_API_KEY")
    
serp_api_key: str = os.getenv("SERP_API_KEY")
    
cloudinary_cloud_name: str = os.getenv("CLOUD_NAME")
cloudinary_api_key: str = os.getenv("API_KEY")
cloudinary_api_secret: str = os.getenv("API_SECRET")
    

