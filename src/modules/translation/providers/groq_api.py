import os
from groq import Groq
from loguru import logger
from ..base import BaseTranslator

class GroqTranslator(BaseTranslator):
    def __init__(self, model_name=None):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = model_name or os.getenv("GROQ_MODEL_ID", "llama-3.1-8b-instant")
        if not self.api_key:
            logger.error("GROQ_API_KEY not found in environment variables.")
        self.client = Groq(api_key=self.api_key)

    def translate(self, messages):
        """
        Messages format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq Translation Error: {e}")
            return None
