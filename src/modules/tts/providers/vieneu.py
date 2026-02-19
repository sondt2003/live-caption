import os
import torch
import numpy as np
from loguru import logger
from ..base import BaseTTS
from utils.utils import save_wav
import soundfile as sf

# Monkey-patch torchao to handle naming conflicts in torchtune
# This avoids the need to modify files in the virtual environment (venv)
try:
    import torchao.quantization.quant_api as quant_api
    if not hasattr(quant_api, 'quantize') and hasattr(quant_api, 'quantize_'):
        quant_api.quantize = quant_api.quantize_
        logger.debug("Monkey-patched torchao.quantization.quant_api.quantize")
except ImportError:
    pass

try:
    # Now import Vieneu - it will use the patched torchao through torchtune
    from vieneu import Vieneu
    VIENEU_AVAILABLE = True
except ImportError:
    VIENEU_AVAILABLE = False
    Vieneu = None

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
            raise ImportError("Thư viện 'vieneu' chưa được cài đặt. Vui lòng cài đặt để sử dụng VieNeu TTS.")

        logger.info(f"Loading VieNeu TTS model: {self.model_id}...")
        try:
            # Use factory function as recommended by author
            self.model = Vieneu(
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
        prompt_text = kwargs.get('prompt_text')
        voice_name = kwargs.get('voice')  # Voice name from engine_run
        voice_dict = kwargs.get('voice_dict') # Full dict override
        
        # Determine reference audio for cloning
        reference_audio = None
        if not voice_dict and not voice_name:
            if speaker_wav and os.path.exists(speaker_wav):
                reference_audio = speaker_wav
                logger.debug(f"Using reference audio for cloning: {reference_audio}")
        
        # Smart Reference Truncation
        # Strategy: VieNeu works best with short references (3-10s).
        # Extended references (>12s) cause hallucinations ("nonsense").
        if reference_audio and os.path.exists(reference_audio):
             try:
                 info = sf.info(reference_audio)
                 if info.duration > 12.0:
                     logger.debug(f"Reference audio too long ({info.duration:.2f}s), truncating to 10s for stability.")
                     # Load and slice
                     audio, sr = sf.read(reference_audio)
                     # Take first 10 seconds
                     max_samples = int(10.0 * sr)
                     if len(audio) > max_samples:
                         audio = audio[:max_samples]
                         # Save to temp file
                         temp_ref = reference_audio.replace('.wav', '_ref_10s.wav')
                         sf.write(temp_ref, audio, sr)
                         reference_audio = temp_ref
                         
                     # Also truncate text to match roughly (assuming ~15cps or just heuristic)
                     # 10s speech ~ 150 chars.
                     if prompt_text and len(prompt_text) > 150:
                         prompt_text = prompt_text[:150]
                         # Try to cut at last space to be cleaner
                         last_space = prompt_text.rfind(' ')
                         if last_space > 100:
                             prompt_text = prompt_text[:last_space]
             except Exception as e:
                 logger.warning(f"Failed to truncate reference audio: {e}")

        # If voice_name is provided (e.g. "Binh"), get the preset dict
        if voice_name and not voice_dict:
            try:
                voice_dict = self.model.get_preset_voice(voice_name)
                logger.info(f"Using VieNeu preset voice: {voice_name}")
            except Exception as e:
                logger.warning(f"Voice preset '{voice_name}' not found, falling back to cloning/default. Error: {e}")

        try:
            # Generate audio (numpy array)
            audio_np = self.model.infer(
                text=text,
                ref_audio=reference_audio,
                ref_text=prompt_text,
                voice=voice_dict
            )
            
            # Save using SDK method (handles sampling rate automatically)
            self.model.save(audio_np, output_path)
            
            if not (os.path.exists(output_path) and os.path.getsize(output_path) > 0):
                raise ValueError("Generated audio file is empty or missing.")
                
        except Exception as e:
            logger.error(f"VieNeu TTS generation failed: {e}")
            # Fallback to silence if error occurs during generation to avoid breaking pipeline
            save_wav(np.zeros(24000), output_path)
