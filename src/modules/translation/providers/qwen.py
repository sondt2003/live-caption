import os
import requests
from ..base import BaseTranslator
from loguru import logger

class QwenTranslator(BaseTranslator):
    def __init__(self):
        self.model_name = os.getenv('QWEN_MODEL_ID', 'qwen-max-2025-01-25')
        self.api_key = os.getenv('QWEN_API_KEY')
        self.base_url = os.getenv('QWEN_API_BASE', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        self.url = f"{self.base_url}/chat/completions"
        self.extra_body = {
            'repetition_penalty': 1.1,
        }

    def translate(self, messages: list, json_mode: bool = True) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "extra_body": self.extra_body
        }
        try:
            response = requests.post(self.url, headers=headers, json=payload, timeout=240)
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"Qwen API Error: {response.status_code} - {response.text}")
                raise Exception(f"Qwen API Error: {response.status_code}")
        except Exception as e:
            logger.error(f"Qwen Translation Error: {e}")
            raise e
