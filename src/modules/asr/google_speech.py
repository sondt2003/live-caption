import os
import json
import requests
import subprocess
import librosa
import numpy as np
from loguru import logger

from concurrent.futures import ThreadPoolExecutor

def _transcribe_segment(i, seg, wav_path, duration, output_dir, lang, api_key):
    start, end = seg["start"], seg["end"]
    # Padding for better perception (0.1s)
    start_pad = max(0, start - 0.1)
    end_pad = min(duration, end + 0.1)
    
    # Use hidden file and put in output folder to avoid CWD clutter
    chunk_path = os.path.join(output_dir, f".temp_vad_chunk_{i}.wav")
    
    # Extract chunk using ffmpeg (precise extraction)
    cmd = [
        'ffmpeg', '-y', '-i', wav_path, 
        '-ss', str(start_pad), '-to', str(end_pad),
        '-ac', '1', '-ar', '16000', '-sample_fmt', 's16',
        chunk_path
    ]
    subprocess.run(cmd, capture_output=True)
    
    # API Request
    url = f"https://www.google.com/speech-api/v2/recognize?output=json&lang={lang}&key={api_key}"
    headers = {'Content-Type': 'audio/l16; rate=16000;'}
    
    segment_transcript = []
    try:
        if not os.path.exists(chunk_path):
            return []
            
        with open(chunk_path, 'rb') as f:
            data = f.read()
        
        response = requests.post(url, headers=headers, data=data)
        content = response.text
        logger.debug(f"Google Response Segment {i}: {content}")
        
        for part in content.split('\n'):
            if not part.strip(): continue
            try:
                data_json = json.loads(part)
                if 'result' in data_json and data_json['result']:
                    for res in data_json['result']:
                        if 'alternative' in res and res['alternative']:
                            text = res['alternative'][0]['transcript']
                            logger.info(f"Phát hiện văn bản (Seg {i}): {text}")
                            segment_transcript.append({
                                "start": start,
                                "end": end,
                                "text": text
                            })
            except Exception as je:
                logger.warning(f"Lỗi parse JSON part: {part}, Lỗi: {je}")
                
    except Exception as e:
        logger.error(f"Google Speech API error at segment {i}: {e}")
    finally:
        if os.path.exists(chunk_path):
            os.remove(chunk_path)
            
    return segment_transcript

def google_transcribe_audio(wav_path, api_key, lang='zh'):
    """
    Transcribe audio using Google Speech API v2 with VAD-based segmenting and Parallel Processing.
    """
    # 1. Load audio once for VAD analysis
    y, sr = librosa.load(wav_path, sr=16000)
    duration = len(y) / sr
    
    # 2. VAD: Split audio based on silence
    # Default is 35dB. Increase (e.g. 40) for less sensitivity (fewer segments), 
    # Decrease (e.g. 25) for more sensitivity.
    vad_db = int(os.getenv('GOOGLE_VAD_DB', 35))
    intervals = librosa.effects.split(y, top_db=vad_db, frame_length=1024, hop_length=256)
    
    # Convert samples to seconds and FILTER out noise (too short segments)
    min_dur = float(os.getenv('GOOGLE_ASR_MIN_DURATION', 0.4))
    segments_to_process = []
    for start_sample, end_sample in intervals:
        s, e = float(start_sample) / sr, float(end_sample) / sr
        if (e - s) >= min_dur:
            segments_to_process.append({"start": s, "end": e})
    
    # Merge segments that are close (e.g. within 0.5s) to provide better context
    merge_gap = float(os.getenv('GOOGLE_ASR_MERGE_GAP', 0.5))
    merged = []
    if segments_to_process:
        curr = segments_to_process[0]
        for next_seg in segments_to_process[1:]:
            if next_seg["start"] - curr["end"] < merge_gap:
                curr["end"] = next_seg["end"]
            else:
                merged.append(curr)
                curr = next_seg
        merged.append(curr)
    
    # 3. Transcribe segments in PARALLEL
    transcript = []
    output_dir = os.path.dirname(wav_path)
    
    # Load concurrency from ENV or default to 5
    max_workers = int(os.getenv('ASR_CONCURRENCY', 5))
    logger.info(f"Dịch vụ nhận diện (Google): Chạy song song {len(merged)} đoạn với {max_workers} luồng...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_transcribe_segment, i, seg, wav_path, duration, output_dir, lang, api_key)
            for i, seg in enumerate(merged)
        ]
        
        for future in futures:
            segment_results = future.result()
            transcript.extend(segment_results)
    
    # IMPORTANT: Sort by start time because futures return in completion order
    transcript.sort(key=lambda x: x['start'])
                
    return transcript
