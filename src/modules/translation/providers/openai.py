import os
from openai import OpenAI
from ..base import BaseTranslator
from loguru import logger

class OpenAITranslator(BaseTranslator):
    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self.model_name = os.getenv('MODEL_NAME', 'gpt-3.5-turbo')
        if 'gpt' not in self.model_name.lower():
            self.model_name = 'gpt-3.5-turbo'
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
            logger.error(f"OpenAI Translation Error: {e}")
            raise e
