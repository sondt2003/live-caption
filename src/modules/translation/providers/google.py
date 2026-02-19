import translators as ts
from ..base import BaseTranslator
from loguru import logger

class GoogleTranslator(BaseTranslator):
    def __init__(self, target_language='vi', server='google'):
        self.target_language = target_language
        self.server = server
        self.lang_map = {
            'vi': 'vi',
            'zh-cn': 'zh-CN',
            'en': 'en',
            'ja': 'ja',
            'ko': 'ko',
            'fr': 'fr',
            'de': 'de',
            'Tiếng Việt': 'vi',
            'Vietnamese': 'vi',
            '中文': 'zh-CN',
            'English': 'en',
            'Japanese': 'ja',
        }

    def _normalize_lang(self, lang):
        return self.lang_map.get(lang.lower(), lang)

    def translate(self, messages: list) -> str:
        # Google/Bing translate works with raw text, not messages list
        # Extract content from the last user message
        text = ""
        for msg in reversed(messages):
            if msg['role'] == 'user':
                # If it's a prompt like Translate:"...", extract it
                content = msg['content']
                if 'Translate:"' in content:
                    text = content.split('Translate:"')[1].split('"')[0]
                else:
                    text = content
                break
        
        if not text:
            return ""

        to_lang = self._normalize_lang(self.target_language)
        
        for retry in range(3):
            try:
                translation = ts.translate_text(
                    query_text=text, 
                    translator=self.server, 
                    from_language='auto', 
                    to_language=to_lang
                )
                return translation
            except Exception as e:
                logger.warning(f"{self.server.capitalize()} Translation failed (retry {retry}): {e}")
        
        return ""
