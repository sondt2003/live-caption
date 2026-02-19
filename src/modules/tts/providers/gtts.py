import os
import torchaudio
from gtts import gTTS
from loguru import logger
from ..base import BaseTTS

class GTTSProvider(BaseTTS):
    def __init__(self):
        # gTTS doesn't need much initialization, but we can set default language mappings if needed
        pass

    def generate(self, text: str, output_path: str, **kwargs) -> None:
        if os.path.exists(output_path):
            return

        target_language = kwargs.get('target_language', 'en')
        
        # Normalize language code for gTTS (e.g., 'zh-cn' -> 'zh-CN')
        lans = target_language.split('-')
        if len(lans) > 1:
            gtts_lang = f"{lans[0]}-{lans[1].upper()}"
        else:
            gtts_lang = target_language

        logger.info(f"Using gTTS for language: {gtts_lang}")
        
        mp3_path = output_path.replace(".wav", ".mp3")
        try:
            tts = gTTS(text, lang=gtts_lang, lang_check=False)
            tts.save(mp3_path)
            
            if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
                # Convert mp3 to wav using torchaudio
                audio, sr = torchaudio.load(mp3_path)
                torchaudio.save(output_path, audio, sr)
                os.remove(mp3_path)
                logger.info(f"gTTS successfully generated and converted to {output_path}")
            else:
                logger.error("gTTS failed to save mp3 file.")
        except Exception as e:
            logger.error(f"gTTS unexpected error: {e}")
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
