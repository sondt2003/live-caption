from loguru import logger

class TTSFactory:
    _instances = {}

    # XTTS supported languages (v2)
    XTTS_LANGS = [
        'en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 
        'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi'
    ]

    @staticmethod
    def get_tts_engine(method):
        method_lower = method.lower()
        
        if method_lower not in TTSFactory._instances:
            if 'edge' in method_lower:
                from .providers.edge import EdgeTTSProvider
                TTSFactory._instances[method_lower] = EdgeTTSProvider()
            elif 'minimax' in method_lower:
                from .providers.minimax import MinimaxProvider
                TTSFactory._instances[method_lower] = MinimaxProvider()
            else:
                # Fallback to EdgeTTS
                logger.warning(f"Unknown TTS method {method}, falling back to EdgeTTS")
                from .providers.edge import EdgeTTSProvider
                TTSFactory._instances[method_lower] = EdgeTTSProvider()
        
        return TTSFactory._instances[method_lower]

    @staticmethod
    def get_best_tts_engine(language: str):
        """
        Automatically selects the best TTS engine based on language.
        Defaults to EdgeTTS for all languages now.
        """
        # Simplification: Always use EdgeTTS as default free provider
        logger.info(f"Auto-selected EdgeTTS for language: {language}")
        return TTSFactory.get_tts_engine('edge')

