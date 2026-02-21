import os
import shutil
import time
from loguru import logger
import subprocess

def extract_audio_from_video(folder, video_path):
    audio_output = os.path.join(folder, 'audio.wav')
    if os.path.exists(audio_output): return audio_output
    cmd = ['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_output]
    subprocess.run(cmd, capture_output=True)
    return audio_output

def separate_audio(folder, model_name="UVR-MDX-NET-Inst_HQ_3"):
    from audio_separator.separator import Separator
    audio_path = os.path.join(folder, 'audio.wav')
    vocal_output_path = os.path.join(folder, 'audio_vocals.wav')
    inst_output_path = os.path.join(folder, 'audio_instruments.wav')
    if os.path.exists(vocal_output_path) and os.path.exists(inst_output_path): return vocal_output_path, inst_output_path
    
    logger.info(f"Separating audio using {model_name}...")
    # Increase batch_size and segment_size to better utilize GPU
    sep = Separator(output_dir=folder, mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 10, "enable_denoise": False})
    sep.load_model(model_name)
    output_files = sep.separate(audio_path)
    if not output_files: raise Exception("Separation failed")
    
    found_v = False
    for f in output_files:
        if "Vocals" in f:
            shutil.move(os.path.join(folder, f), vocal_output_path)
            found_v = True
        elif "Instrumental" in f:
            shutil.move(os.path.join(folder, f), inst_output_path)
    
    if not found_v: raise Exception("Vocals not found after separation")
    if os.path.exists(audio_path): os.remove(audio_path)
    return vocal_output_path, inst_output_path

def separate_all_audio_under_folder(root_folder, model_name="UVR-MDX-NET-Inst_HQ_3", video_path=None):
    vocal = None
    inst = None
    for subdir, dirs, files in os.walk(root_folder):
        if "SPEAKER" in subdir:
            continue
            
        cur_v = video_path if video_path else os.path.join(subdir, 'download.mp4')
        if not os.path.exists(os.path.join(subdir, 'audio_vocals.wav')):
            if not os.path.exists(cur_v):
                continue
            extract_audio_from_video(subdir, cur_v)
            vocal, inst = separate_audio(subdir, model_name)
        else:
            vocal = os.path.join(subdir, 'audio_vocals.wav')
            inst = os.path.join(subdir, 'audio_instruments.wav')
    return 'Done', vocal, inst

def release_model():
    logger.info("Releasing separator model (placeholder)")
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    separate_all_audio_under_folder("outputs")
