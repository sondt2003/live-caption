import json
import time
import librosa
import numpy as np
from faster_whisper import WhisperModel
import os
from loguru import logger
import torch
from dotenv import load_dotenv
load_dotenv()

whisper_model = None
diarize_model = None

def init_whisperx():
    """Initialize faster-whisper model"""
    load_whisper_model()

def init_diarize():
    """Initialize pyannote diarization model"""
    load_diarize_model()

def release_whisperx():
    """
    Release faster-whisper and pyannote models.
    """
    global whisper_model, diarize_model
    logger.info("Releasing ASR models...")
    whisper_model = None
    diarize_model = None
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
def load_whisper_model(model_name: str = 'large', download_root = 'models/ASR/whisper', device='auto'):
    """Load faster-whisper model"""
    if model_name == 'large':
        pretrain_model = os.path.join(download_root,"faster-whisper-large-v3")
        model_name = 'large-v3' if not os.path.isdir(pretrain_model) else pretrain_model
        
    global whisper_model
    if whisper_model is not None:
        return
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Loading faster-whisper model: {model_name}')
    t_start = time.time()
    
    # Use faster-whisper instead of whisperX
    compute_type = "int8" if device == "cpu" else "float16"
    whisper_model = WhisperModel(
        model_name, 
        device=device, 
        compute_type=compute_type,
        download_root=download_root
    )
    t_end = time.time()
    logger.info(f'Loaded faster-whisper model: {model_name} in {t_end - t_start:.2f}s')

def load_diarize_model(device='auto'):
    """Load pyannote speaker diarization model"""
    global diarize_model
    if diarize_model is not None:
        return
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t_start = time.time()
    try:
        from pyannote.audio import Pipeline
        token = os.getenv('HF_TOKEN')
        if not token:
            logger.warning("HF_TOKEN is not set in .env file")
            logger.warning("Please set HF_TOKEN to use speaker diarization")
            return
        else:
            logger.info(f"Using HF_TOKEN (length: {len(token)})")
            
        # Load pyannote speaker-diarization pipeline
        diarize_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token
        )
        
        if device != 'cpu':
            diarize_model.to(torch.device(device))
            
        t_end = time.time()
        logger.info(f'Loaded pyannote diarization model in {t_end - t_start:.2f}s')
    except Exception as e:
        t_end = time.time()
        logger.error(f"Failed to load diarization model: {str(e)}")
        logger.warning("Please ensure you have accepted the terms for 'pyannote/speaker-diarization-3.1' on Hugging Face.")

def whisperx_transcribe_audio(wav_path, model_name: str = 'large', download_root='models/ASR/whisper', device='auto', batch_size=32, diarization=True, min_speakers=None, max_speakers=None):
    """
    Transcribe audio using faster-whisper and optionally apply speaker diarization with pyannote
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load faster-whisper model
    load_whisper_model(model_name, download_root, device)
    
    # Transcribe with faster-whisper
    logger.info(f"Transcribing {wav_path}...")
    t_start = time.time()
    segments, info = whisper_model.transcribe(
        wav_path, 
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    # Convert generator to list and format
    segments_list = []
    for segment in segments:
        segments_list.append({
            'start': segment.start,
            'end': segment.end,
            'text': segment.text.strip()
        })
    
    t_end = time.time()
    logger.info(f"Transcription completed in {t_end - t_start:.2f}s")
    
    if info.language == 'nn' or not segments_list:
        logger.warning(f'No language detected or  no segments in {wav_path}')
        return False
    
    logger.info(f"Detected language: {info.language}")
    
    # Apply speaker diarization if requested
    if diarization:
        load_diarize_model(device)
        if diarize_model:
            logger.info("Running speaker diarization...")
            t_start = time.time()
            try:
                import torchaudio
                waveform, sample_rate = torchaudio.load(wav_path)
                
                # Run diarization
                diarization_kwargs = {"waveform": waveform, "sample_rate": sample_rate}
                if min_speakers and max_speakers:
                    diarization_kwargs["min_speakers"] = min_speakers
                    diarization_kwargs["max_speakers"] = max_speakers
                elif min_speakers:
                    diarization_kwargs["num_speakers"] = min_speakers
                    
                diarization = diarize_model(diarization_kwargs)
                
                # Assign speakers to segments
                for segment in segments_list:
                    segment_start = segment['start']
                    segment_end = segment['end']
                    segment_mid = (segment_start + segment_end) / 2
                    
                    # Find speaker at segment midpoint
                    speaker_at_mid = None
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        if turn.start <= segment_mid <= turn.end:
                            speaker_at_mid = speaker
                            break
                    
                    segment['speaker'] = speaker_at_mid if speaker_at_mid else 'SPEAKER_00'
                
                t_end = time.time()
                logger.info(f"Speaker diarization completed in {t_end - t_start:.2f}s")
            except Exception as e:
                logger.error(f"Diarization failed: {str(e)}")
                # Add default speaker if diarization fails
                for segment in segments_list:
                    segment['speaker'] = 'SPEAKER_00'
        else:
            logger.warning("Diarization model is not loaded, skipping speaker diarization")
            for segment in segments_list:
                segment['speaker'] = 'SPEAKER_00'
    else:
        # No diarization requested
        for segment in segments_list:
            segment['speaker'] = 'SPEAKER_00'
        
    return segments_list


if __name__ == '__main__':
    for root, dirs, files in os.walk("videos"):
        if 'audio_vocals.wav' in files:
            logger.info(f'Transcribing {os.path.join(root, "audio_vocals.wav")}')
            transcript = whisperx_transcribe_audio(os.path.join(root, "audio_vocals.wav"))
            print(transcript)
            break