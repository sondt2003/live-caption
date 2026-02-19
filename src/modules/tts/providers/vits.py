import os
import torch
import numpy as np
from TTS.api import TTS
from huggingface_hub import snapshot_download
from loguru import logger
from ..base import BaseTTS
from utils.utils import save_wav

class VITSProvider(BaseTTS):
    def __init__(self, model_name="JayLL13/VoxCPM-1.5-VN"):
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _init_model(self):
        if self.model is not None:
            return
        
        logger.info(f"Loading VITS model: {self.model_name}...")
        try:
            model_dir = snapshot_download(repo_id=self.model_name)
            checkpoint_path = None
            config_path = os.path.join(model_dir, "config.json")
            
            for f in os.listdir(model_dir):
                if f.endswith(".pth") and "audiovae" not in f and "scheduler" not in f:
                    checkpoint_path = os.path.join(model_dir, f)
                    break
            
            if not checkpoint_path:
                for f in os.listdir(model_dir):
                    if f.endswith(".safetensors"):
                        checkpoint_path = os.path.join(model_dir, f)
                        break

            self.model = TTS(model_path=checkpoint_path, config_path=config_path, progress_bar=False, gpu=(self.device=="cuda"))
            logger.info("VITS model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load VITS model: {e}")
            raise e

    def generate(self, text: str, output_path: str, **kwargs) -> None:
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return

        self._init_model()
        speaker_wav = kwargs.get('speaker_wav')
        target_language = kwargs.get('target_language', 'vi')

        try:
            if speaker_wav and os.path.exists(speaker_wav) and self.model.is_multi_speaker:
                self.model.tts_to_file(text=text, file_path=output_path, speaker_wav=speaker_wav, language=target_language if self.model.is_multi_lingual else None)
            else:
                self.model.tts_to_file(text=text, file_path=output_path, language=target_language if self.model.is_multi_lingual else None)
        except Exception as e:
            logger.error(f"VITS inference failed: {e}")
            save_wav(np.zeros(24000), output_path)
