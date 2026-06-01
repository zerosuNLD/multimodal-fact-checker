import os
from openai import OpenAI
from setting import deepseek_api_key  # Giả định bạn lưu key trong file setting

def call_llm(SYSTEM_PROMPT: str,
            USER_PROMPT: str,
            API_KEY: str = deepseek_api_key,
            MODEL: str = "deepseek-chat",
            temperature: float = 1.0 
                  ) -> str:
    
    client = OpenAI(
        api_key=API_KEY, 
        base_url="https://api.deepseek.com"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            temperature=temperature,
            stream=False
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error: {str(e)}"
