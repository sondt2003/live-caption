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
        
        # Smart Reference Truncation (Dynamic)
        # Strategy: 
        # 1. Calculate rough duration needed for text (char_len / 15 chars per sec).
        # 2. Add a buffer (e.g. +3s).
        # 3. Truncate reference audio to this duration if it exceeds it significantly.
        # 4. Hard limit at 12s to avoid model hallucinations.
        if reference_audio and os.path.exists(reference_audio):
             try:
                 info = sf.info(reference_audio)
                 
                 # Estimate needed duration
                 text_len = len(text) if text else 0
                 # Heuristic: 15 chars per second for Vietnamese speech
                 estimated_dur = (text_len / 15.0) + 3.0 
                 
                 # Clamp estimated duration between 3s and 10s
                 target_dur = max(3.0, min(estimated_dur, 10.0))
                 
                 if info.duration > target_dur + 1.0: # Only truncate if significantly longer
                     logger.debug(f"Reference audio too long ({info.duration:.2f}s) for text ({text_len} chars), truncating to {target_dur:.2f}s.")
                     # Load and slice
                     audio, sr = sf.read(reference_audio)
                     max_samples = int(target_dur * sr)
                     if len(audio) > max_samples:
                         audio = audio[:max_samples]
                         # Save to temp file
                         temp_ref = reference_audio.replace('.wav', f'_ref_{int(target_dur)}s.wav')
                         sf.write(temp_ref, audio, sr)
                         reference_audio = temp_ref
                         
                     # Also truncate prompt text if it seems excessively long compared to the NEW audio
                     # But we prioritize keeping prompt text that matches the truncated audio context
                     # Simple heuristic: 10s audio ~ 150 chars
                     max_prompt_chars = int(target_dur * 18) # slightly more generous
                     if prompt_text and len(prompt_text) > max_prompt_chars:
                          prompt_text = prompt_text[:max_prompt_chars]
                          last_space = prompt_text.rfind(' ')
                          if last_space > (max_prompt_chars - 20):
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
