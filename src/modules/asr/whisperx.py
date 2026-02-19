import json
import time
import librosa
import numpy as np
import whisperx
import os
from loguru import logger
import torch
from dotenv import load_dotenv
load_dotenv()

whisper_model = None
diarize_model = None

align_model = None
language_code = None
align_metadata = None

def init_whisperx():
    """Initialize WhisperX model and alignment model"""
    load_whisper_model()
    # Alignment model will be loaded per-language during transcription
    # but we can pre-load English as a common case if desired.

def init_diarize():
    """Initialize pyannote diarization model (via WhisperX)"""
    load_diarize_model()

def release_whisperx():
    """
    Release models.
    """
    global whisper_model, diarize_model, align_model
    logger.info("Releasing ASR models...")
    whisper_model = None
    diarize_model = None
    align_model = None
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
def load_whisper_model(model_name: str = 'large', download_root = 'models/ASR/whisper', device='auto', language=None):
    """Load WhisperX transcription model"""
    if model_name == 'large':
        pretrain_model = os.path.join(download_root,"faster-whisper-large-v3")
        model_name = 'large-v3' if not os.path.isdir(pretrain_model) else pretrain_model
        
    global whisper_model, language_code
    if whisper_model is not None:
        return
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Loading WhisperX model: {model_name}')
    t_start = time.time()
    
    # Use whisperx.load_model instead of direct faster_whisper
    if device == 'cpu':
        whisper_model = whisperx.load_model(model_name, download_root=download_root, device=device, compute_type='int8', language=language)
    else:
        # Save VRAM by using int8_float16 (reduces usage by ~30%)
        # float16 is faster but uses more VRAM than int8 variants
        whisper_model = whisperx.load_model(model_name, download_root=download_root, device=device, compute_type='int8_float16', language=language)
        
    t_end = time.time()
    logger.info(f'Loaded WhisperX model: {model_name} in {t_end - t_start:.2f}s')

def load_align_model(language='en', device='auto', model_dir='models/ASR/whisper'):
    """Load alignment model for the specific language"""
    global align_model, language_code, align_metadata
    if align_model is not None and language_code == language:
        return
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    language_code = language
    t_start = time.time()
    logger.info(f'Loading alignment model for language: {language_code}')
    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=language_code, device=device, model_dir=model_dir)
        t_end = time.time()
        logger.info(f'Loaded alignment model: {language_code} in {t_end - t_start:.2f}s')
    except Exception as e:
        logger.warning(f"Failed to load alignment model for {language_code}, using unaligned segments. Error: {e}")
        align_model = None

def load_diarize_model(device='auto'):
    """Load pyannote diarization model via WhisperX"""
    global diarize_model
    if diarize_model is not None:
        return
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t_start = time.time()
    try:
        token = os.getenv('HF_TOKEN')
        local_path = os.getenv('DIARIZATION_MODEL_PATH')
        
        # Ưu tiên load từ local path nếu đã download về máy
        if local_path and os.path.exists(local_path):
            logger.info(f"Loading diarization model from local path: {local_path}")
            diarize_model = whisperx.DiarizationPipeline(model_name=local_path, device=device)
        elif token:
            logger.info("Loading diarization model from Hugging Face...")
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=token, device=device)
        else:
            logger.warning("HF_TOKEN and DIARIZATION_MODEL_PATH are not set, skipping diarization")
            return
            
        t_end = time.time()
        logger.info(f'Loaded diarization model in {t_end - t_start:.2f}s')
    except Exception as e:
        t_end = time.time()
        logger.error(f"Failed to load diarization model: {str(e)}")

def whisperx_transcribe_audio(wav_path, model_name: str = 'large', download_root='models/ASR/whisper', device='auto', batch_size=32, diarization=True, min_speakers=None, max_speakers=None, language=None):
    """
    Transcribe audio with alignment and optional diarization.
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Audio to RAM (Optimization: reuse for all stages)
    # This avoids redundant disk I/O and resampling.
    logger.info(f"Loading audio to RAM: {wav_path}")
    audio = whisperx.load_audio(wav_path)
    audio_data = {
        'waveform': torch.from_numpy(audio[None, :]),
        'sample_rate': whisperx.audio.SAMPLE_RATE
    }
    
    # 2. Transcribe
    load_whisper_model(model_name, download_root, device, language=language)
    logger.info(f"Transcribing...")
    t_start = time.time()
    # Use pre-loaded audio for transcription
    result = whisper_model.transcribe(audio, batch_size=batch_size, language=language)
    
    if result['language'] == 'nn' or not result['segments']:
        logger.warning(f'No language detected or no segments in {wav_path}')
        return False
    
    # 3. Align
    load_align_model(result['language'], device, download_root)
    if align_model:
        logger.info(f"Aligning segments for {result['language']}...")
        # Use pre-loaded audio for alignment
        result = whisperx.align(result['segments'], align_model, align_metadata,
                                audio, device, return_char_alignments=False)
    
    # 4. Diarize
    if diarization:
        load_diarize_model(device)
        if diarize_model:
            logger.info("Running speaker diarization...")
            # Optimization: pass pre-loaded audio_data to diarize_model
            # Reference: https://github.com/m-bain/whisperX/issues/399
            diarize_segments = diarize_model(audio_data, min_speakers=min_speakers, max_speakers=max_speakers)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        else:
            logger.warning("Diarization model not available, skipping.")
    
    # Format output
    transcript = []
    for segment in result['segments']:
        transcript.append({
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'].strip(),
            'speaker': segment.get('speaker', 'SPEAKER_00')
        })
    
    t_end = time.time()
    logger.info(f"ASR completed in {t_end - t_start:.2f}s")
    return transcript, audio

if __name__ == '__main__':
    for root, dirs, files in os.walk("videos"):
        if 'audio_vocals.wav' in files:
            logger.info(f'Transcribing {os.path.join(root, "audio_vocals.wav")}')
            transcript = whisperx_transcribe_audio(os.path.join(root, "audio_vocals.wav"))
            print(transcript)
            break