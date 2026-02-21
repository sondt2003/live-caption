import os
import json
import requests
from ..base import BaseTranslator
from loguru import logger

class OllamaTranslator(BaseTranslator):
    def __init__(self):
        self.model_name = os.getenv('OLLAMA_MODEL', 'qwen2.5:14b')
        self.base_url = os.getenv('OLLAMA_API_BASE', 'http://localhost:11434/api')
        self.url = f"{self.base_url}/chat"

    def translate(self, messages: list, json_mode: bool = True) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False
        }
        try:
            logger.info(f"Using Ollama model {self.model_name}...")
            response = requests.post(self.url, json=payload, timeout=120)
            if response.status_code == 200:
                result = response.json()
                return result.get('message', {}).get('content', '')
            else:
                logger.error(f"Ollama API Error: {response.status_code} - {response.text}")
                raise Exception(f"Ollama API Error: {response.status_code}")
        except Exception as e:
            logger.error(f"Ollama Communication Error: {e}")
            raise e
