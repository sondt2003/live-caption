from loguru import logger

class TTSFactory:
    _instances = {}

    @staticmethod
    def get_tts_engine(method):
        method_lower = method.lower()
        
        if method_lower not in TTSFactory._instances:
            if 'edge' in method_lower:
                from .providers.edge import EdgeTTSProvider
                TTSFactory._instances[method_lower] = EdgeTTSProvider()
            elif 'azure' in method_lower:
                from .providers.azure import AzureTTSProvider
                TTSFactory._instances[method_lower] = AzureTTSProvider()
            elif 'openai' in method_lower:
                from .providers.openai_tts import OpenAITTSProvider
                TTSFactory._instances[method_lower] = OpenAITTSProvider()
            elif 'gtts' in method_lower or 'google' in method_lower:
                from .providers.gtts import GTTSProvider
                TTSFactory._instances[method_lower] = GTTSProvider()
            else:
                # Fallback to EdgeTTS
                logger.warning(f"Unknown TTS method {method}, falling back to EdgeTTS")
                from .providers.edge import EdgeTTSProvider
                TTSFactory._instances[method_lower] = EdgeTTSProvider()
        
        return TTSFactory._instances[method_lower]
