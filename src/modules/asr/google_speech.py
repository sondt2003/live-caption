import os
import json
import requests
import subprocess
import librosa
import numpy as np
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

def _transcribe_segment(i, seg, wav_path, duration, output_dir, lang, api_key):
    try:
        import speech_recognition as sr
    except ImportError:
        logger.error("SpeechRecognition library not found. Please install it via pip.")
        return []

    start, end = seg["start"], seg["end"]
    start_pad = max(0, start - 0.1)
    end_pad = min(duration, end + 0.1)
    chunk_path = os.path.join(output_dir, f".temp_vad_chunk_{i}.wav")
    
    # Extract chunk
    cmd = [
        'ffmpeg', '-y', '-i', wav_path, 
        '-ss', str(start_pad), '-to', str(end_pad),
        '-ac', '1', '-ar', '16000', '-sample_fmt', 's16',
        chunk_path
    ]
    subprocess.run(cmd, capture_output=True)
    
    segment_transcript = []
    if not os.path.exists(chunk_path): return []
    
    try:
        r = sr.Recognizer()
        with sr.AudioFile(chunk_path) as source:
            audio_data = r.record(source)
            # Use recognize_google (free web api)
            # If api_key is provided, use it, otherwise None (uses default key)
            key = api_key if api_key and api_key.strip() else None
            text = r.recognize_google(audio_data, key=key, language=lang)
            if text:
                segment_transcript.append({"start": start, "end": end, "text": text})
    except sr.UnknownValueError:
        pass # Audio not clear enough
    except sr.RequestError as e:
        logger.warning(f"Google Speech API request failed: {e}")
    except Exception as e:
        logger.error(f"Error in _transcribe_segment: {e}")
    finally:
        if os.path.exists(chunk_path): os.remove(chunk_path)
        
    return segment_transcript

def google_transcribe_audio(wav_path, api_key, lang='zh'):
    y, sr = librosa.load(wav_path, sr=16000)
    duration = len(y) / sr
    vad_db = int(os.getenv('GOOGLE_VAD_DB', 35))
    intervals = librosa.effects.split(y, top_db=vad_db, frame_length=1024, hop_length=256)
    segments_to_process = []
    for start_sample, end_sample in intervals:
        s, e = float(start_sample) / sr, float(end_sample) / sr
        if (e - s) >= 0.4: segments_to_process.append({"start": s, "end": e})
    merged = []
    if segments_to_process:
        curr = segments_to_process[0]
        for next_seg in segments_to_process[1:]:
            if next_seg["start"] - curr["end"] < 0.5: curr["end"] = next_seg["end"]
            else:
                merged.append(curr)
                curr = next_seg
        merged.append(curr)
    transcript = []
    output_dir = os.path.dirname(wav_path)
    max_workers = int(os.getenv('ASR_CONCURRENCY', 5))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_transcribe_segment, i, seg, wav_path, duration, output_dir, lang, api_key) for i, seg in enumerate(merged)]
        for future in futures: transcript.extend(future.result())
    transcript.sort(key=lambda x: x['start'])
    return transcript
