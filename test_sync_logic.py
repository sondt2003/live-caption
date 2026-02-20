import os
import sys
import numpy as np
import librosa
from loguru import logger

# Mocking parts of the system to test adjust_audio_length
sys.path.append(os.getcwd())
# Ensure src is in path if needed
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.modules.tts.manager import adjust_audio_length

def test_stretch():
    # Create a dummy 4s sine wave
    sr = 24000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration))
    wav = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    test_wav = "/home/dangson/workspace/live-caption/Linly-Dubbing/test_source.wav"
    import soundfile as sf
    sf.write(test_wav, wav, sr)
    
    logger.info(f"Created test wav: {duration}s")
    
    # Try to stretch to 2.0s (speed up 2x)
    desired = 2.0
    stretched_wav, actual_dur = adjust_audio_length(test_wav, desired, sample_rate=sr)
    
    logger.info(f"Target: {desired}s, Actual: {actual_dur:.4f}s")
    
    if abs(actual_dur - desired) < 0.2:
        logger.info("SUCCESS: Audio stretching works!")
    else:
        logger.error(f"FAILURE: Audio stretching failed. Factor logic might be wrong. Ratio needed: {desired/duration}")

    # Test extreme stretch (below min_speed_factor)
    desired_ext = 1.0 # 4x speedup
    stretched_wav, actual_dur_ext = adjust_audio_length(test_wav, desired_ext, sample_rate=sr, min_speed_factor=0.5)
    logger.info(f"Extreme Target: {desired_ext}s, Actual: {actual_dur_ext:.4f}s (Min Speed Factor 0.5 should cap at 2.0s)")

if __name__ == "__main__":
    test_stretch()
