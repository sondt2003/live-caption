import os
from openai import OpenAI
from ..base import BaseTranslator
from loguru import logger

class QwenTranslator(BaseTranslator):
    def __init__(self):
        self.model_name = os.getenv('QWEN_MODEL_ID', 'qwen-max-2025-01-25')
        self.client = OpenAI(
            base_url=os.getenv('QWEN_API_BASE', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
            api_key=os.getenv('QWEN_API_KEY')
        )
        self.extra_body = {
            'repetition_penalty': 1.1,
        }

    def translate(self, messages: list) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                timeout=240,
                extra_body=self.extra_body
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Qwen Translation Error: {e}")
            raise e
