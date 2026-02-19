from .providers.edge import EdgeTTSProvider
from .providers.xtts import XTTSProvider
from .providers.cosyvoice import CosyVoiceProvider
from .providers.vieneu import VieNeuProvider

class TTSFactory:
    _instances = {}

    @staticmethod
    def get_tts_engine(method):
        method_lower = method.lower()
        
        if method_lower not in TTSFactory._instances:
            if 'edge' in method_lower:
                TTSFactory._instances[method_lower] = EdgeTTSProvider()
            elif 'xtts' in method_lower:
                TTSFactory._instances[method_lower] = XTTSProvider()
            elif 'cosyvoice' in method_lower:
                TTSFactory._instances[method_lower] = CosyVoiceProvider()
            elif 'vieneu' in method_lower:
                TTSFactory._instances[method_lower] = VieNeuProvider()
            else:
                TTSFactory._instances[method_lower] = EdgeTTSProvider()
        
        return TTSFactory._instances[method_lower]
