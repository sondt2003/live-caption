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
            elif 'xtts' in method_lower:
                from .providers.xtts import XTTSProvider
                TTSFactory._instances[method_lower] = XTTSProvider()
            elif 'vieneu' in method_lower:
                from .providers.vieneu import VieNeuProvider
                TTSFactory._instances[method_lower] = VieNeuProvider()
            elif 'elevenlabs' in method_lower or 'ai33' in method_lower:
                from .providers.elevenlabs import ElevenLabsProvider
                TTSFactory._instances[method_lower] = ElevenLabsProvider()
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
        Prioritizes XTTS for its supported languages.
        """
        lang_lower = language.lower()
        if lang_lower == 'zh': lang_lower = 'zh-cn'
        
        if lang_lower == 'vi':
            logger.info(f"Auto-selected VieNeu-TTS for language: {language}")
            return TTSFactory.get_tts_engine('vieneu')
        elif lang_lower in TTSFactory.XTTS_LANGS:
            logger.info(f"Auto-selected XTTS for language: {language}")
            return TTSFactory.get_tts_engine('xtts')
        else:
            logger.info(f"Auto-selected EdgeTTS for language: {language}")
            return TTSFactory.get_tts_engine('edge')

