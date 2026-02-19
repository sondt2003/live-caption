import os
import torch
import numpy as np
from loguru import logger
from ..base import BaseTTS
from utils.utils import save_wav

try:
    from vieneu import VieNeuTTS
    import soundfile as sf
    VIENEU_AVAILABLE = True
except ImportError:
    VIENEU_AVAILABLE = False
    VieNeuTTS = None
    sf = None

class VieNeuProvider(BaseTTS):
    """
    Vietnamese TTS Provider using VieNeu model for high-quality voice cloning.
    Model Repo: pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf
    """
    def __init__(self, model_id="pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf"):
        self.model_id = model_id
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _init_model(self):
        if self.model is not None:
            return
        
        if not VIENEU_AVAILABLE:
            raise ImportError("Thư viện 'vieneu' hoặc 'soundfile' chưa được cài đặt. Vui lòng cài đặt để sử dụng VieNeu TTS.")

        logger.info(f"Loading VieNeu TTS model: {self.model_id}...")
        try:
            self.model = VieNeuTTS(
                backbone_repo=self.model_id,
                backbone_device=self.device,
                codec_device=self.device
            )
            logger.info("VieNeu TTS model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load VieNeu TTS model: {e}")
            raise e

    def generate(self, text: str, output_path: str, **kwargs) -> None:
        """
        Generate audio from text using VieNeu.
        Supports instant voice cloning via 'speaker_wav' argument.
        """
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return

        self._init_model()
        speaker_wav = kwargs.get('speaker_wav')
        
        # Determine reference audio for cloning
        reference_audio = None
        if speaker_wav and os.path.exists(speaker_wav):
            reference_audio = speaker_wav
            logger.debug(f"Using reference audio for cloning: {reference_audio}")
        
        try:
            # Generate audio (numpy array)
            audio_np = self.model.infer(
                text=text,
                ref_audio=reference_audio
            )
            
            # Save using soundfile (sampling rate is 24000)
            sf.write(output_path, audio_np, 24000)
            
            if not (os.path.exists(output_path) and os.path.getsize(output_path) > 0):
                raise ValueError("Generated audio file is empty or missing.")
                
        except Exception as e:
            logger.error(f"VieNeu TTS generation failed: {e}")
            # Fallback to silence if error occurs during generation to avoid breaking pipeline
            save_wav(np.zeros(24000), output_path)
