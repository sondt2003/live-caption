import os
import torch
import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download
from loguru import logger
from ..base import BaseTTS
from utils.utils import save_wav

try:
    from voxcpm.core import VoxCPM
    VOXCPM_AVAILABLE = True
except ImportError:
    VOXCPM_AVAILABLE = False
    logger.warning("voxcpm library not found. Please install it with 'pip install voxcpm'")

class VoxCPMProvider(BaseTTS):
    def __init__(self, model_id="JayLL13/VoxCPM-1.5-VN"):
        self.model_id = model_id
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _init_model(self):
        if not VOXCPM_AVAILABLE:
            raise ImportError("voxcpm library is required for VoxCPMProvider. Install it via 'pip install voxcpm'")
            
        if self.model is not None:
            return
        
        logger.info(f"Loading VoxCPM model: {self.model_id}...")
        try:
            # Check if model_id is a local path or HF repo
            if os.path.exists(self.model_id):
                model_path = self.model_id
            else:
                model_path = snapshot_download(repo_id=self.model_id)
            
            # Use optimize=False by default to avoid torch.compile issues in some environments
            self.model = VoxCPM.from_pretrained(
                hf_model_id=model_path,
                load_denoiser=False,
                optimize=False,
            )
            logger.info("VoxCPM model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load VoxCPM model: {e}")
            raise e

    def generate(self, text: str, output_path: str, **kwargs) -> None:
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return

        self._init_model()
        prompt_wav_path = kwargs.get('speaker_wav')
        prompt_text = kwargs.get('prompt_text')
        
        # VoxCPM specific params
        cfg_value = kwargs.get('cfg_value', 2.0)
        inference_timesteps = kwargs.get('inference_timesteps', 10)
        
        try:
            logger.info(f"VoxCPM generating: '{text}'")
            audio_np = self.model.generate(
                text=text,
                prompt_wav_path=prompt_wav_path if prompt_wav_path and os.path.exists(prompt_wav_path) else None,
                prompt_text=prompt_text,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                denoise=False,
                normalize=True
            )
            
            # VoxCPM output is numpy array
            save_wav(audio_np, output_path, sample_rate=self.model.tts_model.sample_rate)
            
            # Explicit cleanup after generation
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        except Exception as e:
            logger.error(f"VoxCPM inference failed: {e}")
            # Save silence on failure
            save_wav(np.zeros(24000), output_path)

    def release(self):
        """
        Release the model and free up VRAM.
        """
        if self.model is not None:
            logger.info("Releasing VoxCPM model...")
            self.model = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
