
import os
import sys
import torch
import numpy as np
from TTS.api import TTS
from loguru import logger
from ..utils.utils import save_wav

# Global model instance
vits_model = None

def load_vits_model(model_name="JayLL13/VoxCPM-1.5-VN", device='auto'):
    global vits_model
    if vits_model is not None:
        return

    if device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading VITS model: {model_name} on {device}...")
    
    try:
        # TTS api handles downloading and loading from HF hub automatically
        # For VoxCPM, we might need to be specific if it's not a standard Coqui model
        # But based on tests, it seems compatible with TTS(model_path=...) or model_name if registered
        # Since it's a HF model, we can try passing the repo ID directly
        
        # Note: If the previous download was incomplete, this might fail or try to resume.
        # We assume the model is available or will be downloaded.
        
        # We need to point to the specific files if generic load fails, but let's try generic first.
        # Actually, for VITS models on HF that are not official Coqui models, we usually need to download snapshot 
        # and point to the files.
        
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(repo_id=model_name)
        
        # Find checkpoint and config
        checkpoint_path = None
        config_path = os.path.join(model_dir, "config.json")
        
        files = os.listdir(model_dir)
        for f in files:
            if f.endswith(".pth") and "audiovae" not in f and "scheduler" not in f:
                checkpoint_path = os.path.join(model_dir, f)
                break
        
        if not checkpoint_path:
             for f in files:
                if f.endswith(".safetensors"):
                    checkpoint_path = os.path.join(model_dir, f)
                    break
        
        if not checkpoint_path:
            raise FileNotFoundError(f"No checkpoint found in {model_dir}")

        vits_model = TTS(model_path=checkpoint_path, config_path=config_path, progress_bar=False, gpu=(device=="cuda"))
        logger.info("VITS model loaded successfully.")
        
    except Exception as e:
        logger.error(f"Failed to load VITS model: {e}")
        raise e

def tts(text, output_path, speaker_wav=None, device='auto', target_language='vi'):
    global vits_model
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return

    if vits_model is None:
        load_vits_model(device=device)
        
    logger.info(f"VITS TTS: {text}")
    try:
        # Check if model supports Speaker cloning
        # VoxCPM usually does if it's VITS
        
        # If speaker_wav is provided and model is multi-speaker, use it.
        # TTS API handles this: if speaker_wav is passed but model is single speaker, it might ignore or error.
        # We'll try with speaker_wav if provided.
        
        if speaker_wav and os.path.exists(speaker_wav) and vits_model.is_multi_speaker:
             vits_model.tts_to_file(text=text, file_path=output_path, speaker_wav=speaker_wav, language=target_language if vits_model.is_multi_lingual else None)
        else:
             # Fallback or single speaker
             vits_model.tts_to_file(text=text, file_path=output_path, language=target_language if vits_model.is_multi_lingual else None)
             
    except Exception as e:
        logger.error(f"VITS inference failed: {e}")
        # Create silent file fallback? Or let it fail to be noticed?
        # Creating a fallback silence is safer for pipeline
        logger.warning("Generating silence due to error.")
        save_wav(np.zeros(24000), output_path)

