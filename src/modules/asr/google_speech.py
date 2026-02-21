import os
import json
import wave
import requests
import subprocess
import soundfile as sf
import numpy as np
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

_FALLBACK_KEY = 'AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw'
_GOOGLE_ASR_URL = "https://www.google.com/speech-api/v2/recognize?output=json&lang={lang}&key={key}"


def _transcribe_segment(i: int, seg: dict, wav_path: str, duration: float,
                        output_dir: str, lang: str, api_key: str) -> list:
    start, end = seg["start"], seg["end"]
    chunk_path = os.path.join(output_dir, f".temp_chunk_{i}.wav")

    subprocess.run(
        ['ffmpeg', '-y', '-i', wav_path,
         '-ss', str(start), '-to', str(end),
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
        logger.debug(f"[ASR Seg {i}] {start:.1f}s-{end:.1f}s HTTP {resp.status_code}")

        for line in resp.text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                for alt_block in obj.get('result', []):
                    alts = alt_block.get('alternative', [])
                    if alts:
                        text = alts[0].get('transcript', '').strip()
                        if text:
                            logger.info(f"[ASR Seg {i}] Phát hiện văn bản: {text[:70]}")
                            results.append({"start": start, "end": end, "text": text})
            except json.JSONDecodeError:
                pass
    except Exception as e:
        logger.error(f"[ASR Seg {i}] Error: {e}")
    finally:
        try:
            os.remove(chunk_path)
        except OSError:
            pass

    return results


def google_transcribe_audio(wav_path: str, api_key: str, lang: str = 'zh') -> list:
    """
    Transcribe audio using Google Speech API v2.
    Splits audio into fixed-size chunks and transcribes in parallel.
    No VAD — uses GOOGLE_MAX_CHUNK_SEC (default 30s) to split.
    """
    y, sr = sf.read(wav_path, dtype='float32', always_2d=False)
    duration = float(len(y)) / sr

    chunk_size  = float(os.getenv('GOOGLE_MAX_CHUNK_SEC', 30))
    max_workers = int(os.getenv('ASR_CONCURRENCY', 5))

    # Split into fixed chunks
    chunks = []
    start = 0.0
    while start < duration:
        end = min(start + chunk_size, duration)
        if end - start >= 0.5:  # skip tiny trailing slices
            chunks.append({"start": start, "end": end})
        start = end

    logger.info(f"[ASR] Google Speech: {len(chunks)} chunks (≤{chunk_size}s), {max_workers} workers")

    output_dir = os.path.dirname(wav_path) or '.'
    ordered = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_transcribe_segment, i, seg, wav_path, duration, output_dir, lang, api_key): i
            for i, seg in enumerate(chunks)
        }
        for future in future_map:
            idx = future_map[future]
            ordered[idx] = future.result()

    transcript = []
    for segs in ordered:
        if segs:
            transcript.extend(segs)

    transcript.sort(key=lambda x: x['start'])
    return transcript
