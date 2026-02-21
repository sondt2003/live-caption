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

    def translate(self, messages, json_mode=True, max_tokens=None):
        """
        Messages format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        if not self.api_key:
            return None
            
        import time
        max_retries = 3
        retry_delay = 2 # initial delay in seconds
        
        for attempt in range(max_retries):
            try:
                # Default max_tokens increased to 4096 to prevent JSON truncation
                # especially for long translations or verbose model behaviors.
                limit = max_tokens or int(os.getenv("GROQ_MAX_TOKENS", 4096))
                
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": limit,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                    
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except Exception as e:
                # Check for rate limit error (429)
                if "429" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"Groq Rate Limit exceeded (429). Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2 # Exponential backoff
                    continue
                
                logger.error(f"Groq Translation Error: {e}")
                return None
