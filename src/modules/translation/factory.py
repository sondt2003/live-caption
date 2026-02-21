from .providers.ollama import OllamaTranslator
from .providers.qwen import QwenTranslator
from .providers.ernie import ErnieTranslator
from .providers.google import GoogleTranslator
from .providers.groq_api import GroqTranslator
from .providers.llm import LocalLLMTranslator
from .providers.gemini import GeminiTranslator

class TranslatorFactory:
    @staticmethod
    def get_translator(method, target_language='vi'):
        method_lower = method.lower()
        
        if 'ollama' in method_lower:
            return OllamaTranslator()
        elif 'qwen' in method_lower or '通义千问' in method:
            return QwenTranslator()
        elif 'ernie' in method_lower or 'baidu' in method_lower:
            return ErnieTranslator()
        elif 'gemini' in method_lower:
            return GeminiTranslator()
        elif 'google' in method_lower:
            return GoogleTranslator(target_language=target_language, server='google')
        elif 'groq' in method_lower:
            return GroqTranslator()
        elif 'bing' in method_lower:
            return GoogleTranslator(target_language=target_language, server='bing')
        elif 'llm' in method_lower:
            return LocalLLMTranslator()
        else:
            return GoogleTranslator(target_language=target_language, server='google')
