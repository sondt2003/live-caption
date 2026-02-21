import os
import json
import wave
import torch
import requests
import subprocess
import soundfile as sf
import numpy as np
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

_FALLBACK_KEY = 'AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw'
_GOOGLE_ASR_URL = "https://www.google.com/speech-api/v2/recognize?output=json&lang={lang}&key={key}"

# ── Silero VAD (same as WhisperX) ────────────────────────────────────────────
_silero_model = None

def _get_silero():
    global _silero_model
    if _silero_model is None:
        logger.info("Loading Silero VAD model (WhisperX style)...")
        _silero_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', model='silero_vad',
            force_reload=False, onnx=False, trust_repo=True)
        _silero_model._get_ts = utils[0]   # get_speech_timestamps
    return _silero_model

def _silero_segments(y: np.ndarray, chunk_size: float, onset: float) -> list:
    """
    Run Silero VAD — mirrors WhisperX Silero class exactly:
      get_speech_timestamps(waveform, max_speech_duration_s=chunk_size, threshold=onset)
    Then merge with WhisperX Vad.merge_chunks(chunk_size=chunk_size) logic.
    """
    model = _get_silero()
    waveform = torch.from_numpy(y).float()
    
    # WhisperX uses 16000Hz fixed for Silero
    stamps = model._get_ts(
        waveform, model=model, sampling_rate=16000,
        max_speech_duration_s=chunk_size,
        threshold=onset,
    )
    raw = [{"start": s["start"] / 16000, "end": s["end"] / 16000} for s in stamps]
    if not raw:
        return []

    # WhisperX Vad.merge_chunks logic (vads/vad.py)
    merged = []
    curr_start = raw[0]["start"]
    curr_end = 0.0
    for seg in raw:
        # If adding this segment exceeds chunk_size, close current and start new
        if seg["end"] - curr_start > chunk_size and curr_end - curr_start > 0:
            merged.append({"start": curr_start, "end": curr_end})
            curr_start = seg["start"]
        curr_end = seg["end"]
    
    # Add final segment
    merged.append({"start": curr_start, "end": curr_end})
    return merged


# ── Transcribe one chunk via Google Speech API v2 ────────────────────────────

def _transcribe_segment(i: int, seg: dict, wav_path: str, duration: float,
                        output_dir: str, lang: str, api_key: str) -> list:
    start, end = seg["start"], seg["end"]
    chunk_path = os.path.join(output_dir, f".temp_chunk_{i}.wav")

    # Extract chunk with small pads to avoid cutting off words
    start_pad = max(0, start - 0.05)
    end_pad = min(duration, end + 0.05)

    subprocess.run(
        ['ffmpeg', '-y', '-i', wav_path,
         '-ss', str(start_pad), '-to', str(end_pad),
         '-ac', '1', '-ar', '16000', '-sample_fmt', 's16', chunk_path],
        capture_output=True
    )
    if not os.path.exists(chunk_path):
        return []

    key = api_key if api_key and str(api_key).strip() not in ('', 'None') else _FALLBACK_KEY
    url = _GOOGLE_ASR_URL.format(lang=lang, key=key)
    headers = {'Content-Type': 'audio/l16; rate=16000;'}

    results = []
    try:
        with wave.open(chunk_path, 'rb') as wf:
            pcm = wf.readframes(wf.getnframes())
        resp = requests.post(url, headers=headers, data=pcm, timeout=30)
        
        # Google returns multiple JSON lines
        for line in resp.text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                for blk in obj.get('result', []):
                    alts = blk.get('alternative', [])
                    if alts:
                        text = alts[0].get('transcript', '').strip()
                        if text:
                            logger.info(f"[ASR Seg {i}] ({start:.1f}s) Phát hiện: {text[:70]}")
                            # Keep original seg start/end for alignment
                            results.append({"start": start, "end": end, "text": text})
            except json.JSONDecodeError:
                pass
    except Exception as e:
        logger.error(f"[ASR Seg {i}] Error: {e}")
    finally:
        if os.path.exists(chunk_path):
            try: os.remove(chunk_path)
            except: pass
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def google_transcribe_audio(wav_path: str, api_key: str, lang: str = 'zh') -> list:
    """
    Transcribe via Google Speech API v2 using Silero VAD (WhisperX style).
    """
    y, sr = sf.read(wav_path, dtype='float32', always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    duration = float(len(y)) / 16000

    # Match WhisperX defaults
    chunk_size  = float(os.getenv('GOOGLE_MAX_CHUNK_SEC', 30))
    vad_onset   = 0.500
    max_workers = int(os.getenv('ASR_CONCURRENCY', 5))

    # 1. Silero VAD -> merged chunks <= 30s
    try:
        chunks = _silero_segments(y, chunk_size=chunk_size, onset=vad_onset)
        logger.info(f"[ASR] Silero VAD → {len(chunks)} chunks (WhisperX style, max={chunk_size}s)")
    except Exception as e:
        logger.warning(f"[ASR] Silero failed ({e}), fallback to fixed 10s chunks")
        chunks = []
        start = 0.0
        while start < duration:
            end = min(start + 10.0, duration)
            if end - start >= 0.5:
                chunks.append({"start": start, "end": end})
            start = end

    if not chunks:
        return []

    # 2. Parallel transcription
    output_dir = os.path.dirname(wav_path) or '.'
    ordered = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_transcribe_segment, i, seg, wav_path, duration, output_dir, lang, api_key): i
            for i, seg in enumerate(chunks)
        }
        for future in future_map:
            ordered[future_map[future]] = future.result()

    transcript = []
    for seg_results in ordered:
        if seg_results:
            transcript.extend(seg_results)
    
    transcript.sort(key=lambda x: x['start'])
    return transcript
