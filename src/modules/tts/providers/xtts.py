import os
import time
import torch
import numpy as np
from loguru import logger
try:
    from TTS.api import TTS
except Exception as e:
    logger.warning(f"Failed to import TTS (Coqui TTS). XTTS will not be available. Error: {e}")
    TTS = None

from ..base import BaseTTS
from utils.utils import save_wav

class XTTSProvider(BaseTTS):
    def __init__(self, model_path="tts_models/multilingual/multi-dataset/xtts_v2"):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.language_map = {
            '中文': 'zh-cn', '简体中文': 'zh-cn', 'English': 'en', 'Spanish': 'es',
            'French': 'fr', 'German': 'de', 'Italian': 'it', 'Portuguese': 'pt',
            'Polish': 'pl', 'Turkish': 'tr', 'Russian': 'ru', 'Dutch': 'nl',
            'Czech': 'cs', 'Arabic': 'ar', 'Hungarian': 'hu', 'Hindi': 'hi',
            'Korean': 'ko', 'Japanese': 'ja',
        }

    def _init_model(self):
        if self.model is not None:
            return
        
        if TTS is None:
            raise ImportError("Thư viện 'coqui-tts' chưa được cài đặt hoặc bị lỗi. Không thể sử dụng XTTS.")

        logger.info(f"Loading XTTS model: {self.model_path}...")
        try:
            if os.path.exists(self.model_path):
                self.model = TTS(model_path=self.model_path, config_path=os.path.join(self.model_path, 'config.json')).to(self.device)
            else:
                self.model = TTS(self.model_path).to(self.device)
        except Exception as e:
            logger.warning(f"Fallback loading XTTS: {e}")
            try:
                self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            except Exception as e2:
                logger.error(f"Failed to load XTTS model: {e2}")
                raise e2

    def generate(self, text: str, output_path: str, **kwargs) -> None:
        if os.path.exists(output_path):
            return

        self._init_model()
        target_language = kwargs.get('target_language', 'en')
        speaker_wav = kwargs.get('speaker_wav')

        # Language mapping logic
        language = self.language_map.get(target_language, target_language if target_language in self.language_map.values() else 'en')
        if language == 'vi': language = 'en' # XTTS doesn't support vi

        for retry in range(3):
            try:
                wav = self.model.tts(text, speaker_wav=speaker_wav, language=language)
                save_wav(np.array(wav), output_path)
                break
            except Exception as e:
                logger.error(f"XTTS error (retry {retry}): {e}")
