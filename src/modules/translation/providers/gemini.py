import os
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import google.generativeai as genai
from loguru import logger
from ..base import BaseTranslator

class GeminiTranslator(BaseTranslator):
    def __init__(self, model_name="gemini-1.5-flash"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        if not self.api_key:
            logger.error("GOOGLE_API_KEY not found in environment variables.")
        else:
            genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def translate(self, messages: list, json_mode: bool = True) -> str:
        """
        Messages format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        if not self.api_key:
            logger.error("Gemini Translation failed: GOOGLE_API_KEY missing.")
            return None

        prompt = ""
        for msg in messages:
            role = "Context" if msg['role'] == 'system' else "User"
            prompt += f"{role}: {msg['content']}\n"
        
        try:
            generation_config = {
                "temperature": 0.1,
            }
            if json_mode:
                generation_config["response_mime_type"] = "application/json"

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini Translation Error: {e}")
            return None
