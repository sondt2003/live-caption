import os
import sys
import time
import torch
import torchaudio
from loguru import logger
from ..base import BaseTTS
from modelscope import snapshot_download

class CosyVoiceProvider(BaseTTS):
    def __init__(self, model_path="models/TTS/CosyVoice-300M"):
        self.model_path = model_path
        self.model = None
        self.language_map = {
            '中文': 'zh', 'English': 'en', 'Japanese': 'jp', '粤语': 'yue', 'Korean': 'ko'
        }

    def _init_model(self):
        if self.model is not None:
            return
        
        # Add CosyVoice paths to sys.path
        if 'CosyVoice' not in sys.path:
            sys.path.append('CosyVoice/third_party/Matcha-TTS')
            sys.path.append('CosyVoice/')
        
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice
        except ImportError as e:
            logger.warning(f"Failed to import CosyVoice. CosyVoice will not be available. Error: {e}")
            self.model = None
            return

        
        if not os.path.exists(self.model_path):
            logger.info("Downloading CosyVoice model...")
            snapshot_download('iic/CosyVoice-300M', local_dir=self.model_path)
        
        logger.info(f"Loading CosyVoice model from {self.model_path}...")
        self.model = CosyVoice(self.model_path)

    def generate(self, text: str, output_path: str, **kwargs) -> None:
        if os.path.exists(output_path):
            return

        self._init_model()
        speaker_wav = kwargs.get('speaker_wav')
        target_language = kwargs.get('target_language', '中文')
        
        from cosyvoice.utils.file_utils import load_wav
        
        lang_code = self.language_map.get(target_language, 'zh')
        for retry in range(3):
            try:
                prompt_speech_16k = load_wav(speaker_wav, 16000)
                output = self.model.inference_cross_lingual(f'<|{lang_code}|>{text}', prompt_speech_16k)
                torchaudio.save(output_path, output['tts_speech'], 22050)
                break
            except Exception as e:
                logger.error(f"CosyVoice error (retry {retry}): {e}")
