import os
import torch
import torchaudio
from df.enhance import enhance, init_df, load_audio, save_audio
from loguru import logger
import time

# Global variables for model state
df_model = None
df_state = None

def init_dereverb():
    """
    Initialize DeepFilterNet model.
    """
    global df_model, df_state
    if df_model is None:
        logger.info("Initializing DeepFilterNet for dereverb...")
        t_start = time.time()
        # init_df will download the model to ~/.cache/deepfilternet if not present
        df_model, df_state, _ = init_df()
        t_end = time.time()
        logger.info(f"DeepFilterNet initialized in {t_end - t_start:.2f}s")

def release_dereverb():
    """
    Release DeepFilterNet model resources.
    """
    global df_model, df_state
    if df_model is not None:
        logger.info("Releasing DeepFilterNet model...")
        df_model = None
        df_state = None
        torch.cuda.empty_cache()
        import gc
        gc.collect()

def dereverb_audio(input_path: str, output_path: str = None):
    """
    Apply dereverberation to an audio file using DeepFilterNet.
    """
    global df_model, df_state
    
    if output_path is None:
        output_path = input_path.replace('.wav', '_dereverb.wav')
        
    if os.path.exists(output_path):
        logger.info(f"Dereverb audio already exists: {output_path}")
        return output_path

    if df_model is None:
        init_dereverb()

    logger.info(f"Applying dereverb to {input_path}...")
    t_start = time.time()
    try:
        # DeepFilterNet expects 48kHz by default for the latest models
        audio, _ = load_audio(input_path, sr=df_state.sr())
        enhanced = enhance(df_model, df_state, audio)
        save_audio(output_path, enhanced, df_state.sr())
        t_end = time.time()
        logger.info(f"Dereverb completed in {t_end - t_start:.2f}s. Saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Dereverb failed: {e}")
        return input_path # Fallback to original

def process_folder_dereverb(folder: str):
    """
    Dereverb the vocals in a specific folder.
    """
    vocal_path = os.path.join(folder, 'audio_vocals.wav')
    if os.path.exists(vocal_path):
        dereverb_path = os.path.join(folder, 'audio_vocals_dereverb.wav')
        return dereverb_audio(vocal_path, dereverb_path)
    return None

if __name__ == '__main__':
    # Test on a specific folder if needed
    test_folder = "compare_results/temp/video3"
    if os.path.exists(test_folder):
        process_folder_dereverb(test_folder)
