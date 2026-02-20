import json
import os
import re
import librosa
import shutil
import numpy as np
import pyloudnorm as pyln
from loguru import logger
from audiostretchy.stretch import stretch_audio

from utils.utils import save_wav, save_wav_norm
from .factory import TTSFactory

def preprocess_text(text, target_language='vi'):
    if 'zh-cn' in target_language.lower():
        text = text.replace('AI', '人工智能')
        # Add basic spaces between English and Chinese if needed
        text = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)
    else:
        # Generic cleanup for non-Chinese languages
        text = text.replace('AI', 'A I') # Pronounce letters for ASR/TTS
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
    return text

def adjust_audio_length(wav_path, desired_length, sample_rate=24000, min_speed_factor=0.6, max_speed_factor=1.2):
    try:
        # Load to check length
        wav_orig, _ = librosa.load(wav_path, sr=sample_rate)
        current_length = len(wav_orig)/sample_rate
    except Exception as e:
        logger.warning(f"Audio load failed, returning silence: {e}")
        return np.zeros((int(desired_length * sample_rate), )), desired_length
    
    # Calculate speed factor: ratio = target / source
    # If ratio < 1.0 -> speed up
    # If ratio > 1.0 -> slow down
    speed_factor = max(min(desired_length / current_length, max_speed_factor), min_speed_factor)
    
    target_path = wav_path.replace('.wav', '_adjusted.wav').replace('.mp3', '_adjusted.wav')
    stretch_audio(wav_path, target_path, ratio=speed_factor, sample_rate=sample_rate)
    
    # Load the actual stretched audio
    wav, _ = librosa.load(target_path, sr=sample_rate)
    actual_duration = len(wav) / sample_rate
    
    return wav, actual_duration

def generate_all_wavs_under_folder(folder, method='auto', target_language='vi', voice='vi-VN-HoaiMyNeural', video_volume=1.0):
    transcript_path = os.path.join(folder, 'translation.json')
    output_folder = os.path.join(folder, 'wavs')
    os.makedirs(output_folder, exist_ok=True)

    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    # Normalize language code
    target_language = target_language.lower().replace('tiếng việt', 'vi').replace('简体中文', 'zh-cn')
    
    # Khởi tạo engine TTS từ Factory
    if method is None or method.lower() == 'auto':
        engine = TTSFactory.get_best_tts_engine(target_language)
    else:
        engine = TTSFactory.get_tts_engine(method)

    
    # Collect all tasks for batch processing
    tasks = []
    for i, line in enumerate(transcript):
        speaker = line['speaker']
        text = preprocess_text(line['translation'], target_language)
        output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        
        # Logic tìm kiếm giọng tham chiếu (Reference Audio)
        # Ưu tiên các file cắt riêng cho từng speaker (ngắn và tập trung hơn)
        speaker_wav = os.path.join(folder, 'SPEAKER', f'{speaker}.wav')
        if not os.path.exists(speaker_wav):
            vocal_dereverb_wav = os.path.join(folder, 'audio_vocals_dereverb.wav')
            speaker_wav = vocal_dereverb_wav if os.path.exists(vocal_dereverb_wav) else os.path.join(folder, 'audio_vocals.wav')

        # VieNeu cloning works better without ref_text if reference is cross-language
        tasks.append({
            "text": text,
            "output_path": output_path,
            "speaker_wav": speaker_wav,
            "ref_text": None, # Skip ref_text to avoid alignment issues/token overflow
            "target_language": target_language,
            "voice": voice
        })

    # Gọi Engine để tạo âm thanh hàng loạt (Batch Processing)
    if hasattr(engine, 'generate_batch'):
        engine.generate_batch(tasks)
    else:
        # Fallback for engines that don't support batching
        for task in tasks:
            engine.generate(task.pop("text"), task.pop("output_path"), **task)

    full_wav = np.zeros((0, ))
    for i, line in enumerate(transcript):
        output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        start, end = line['start'], line['end']
        # Backup original timings for visual sync
        line['original_start'] = start
        line['original_end'] = end
        
        length = end - start
        last_end = len(full_wav)/24000
        
        if start > last_end:
            # Thêm khoảng lặng nếu câu tiếp theo bắt đầu muộn hơn câu trước
            full_wav = np.concatenate((full_wav, np.zeros((int((start - last_end) * 24000), ))))
        
        current_start = len(full_wav)/24000
        line['start'] = current_start
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            # Preserve source duration for video synchronization
            line['source_duration'] = length
            # Điều chỉnh độ dài âm thanh để khớp với Timeline
            wav, adjusted_len = adjust_audio_length(output_path, length)
        else:
            line['source_duration'] = length
            wav = np.zeros((int(length * 24000), ))
            adjusted_len = length

        full_wav = np.concatenate((full_wav, wav))
        line['end'] = current_start + adjusted_len

    # Lưu lại transcript đã cập nhật với thông tin đồng bộ thích ứng
    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    # Xử lý âm thanh hậu kỳ (Mastering)
    logger.info("Đang xử lý âm thanh hậu kỳ (Studio-Grade Mastering)...")
    try:
        meter = pyln.Meter(24000)
        loudness = meter.integrated_loudness(full_wav)
        if np.isinf(loudness):
            logger.warning("Độ lớn âm thanh không xác định (-inf), bỏ qua chuẩn hóa.")
        else:
            full_wav = pyln.normalize.loudness(full_wav, loudness, -23.0)
    except Exception as e:
        logger.warning(f"Mastering thất bại: {e}")

    save_wav(full_wav, os.path.join(folder, 'audio_tts.wav'))
    
    # Trộn với nhạc nền/âm thanh gốc nếu có
    instr_path = os.path.join(folder, 'audio_instruments.wav')
    if os.path.exists(instr_path):
        instr, _ = librosa.load(instr_path, sr=24000)
        min_len = min(len(full_wav), len(instr))
        # Áp dụng video_volume để giảm tiếng gốc nếu cần
        combined = full_wav[:min_len] + instr[:min_len] * video_volume
        save_wav_norm(combined, os.path.join(folder, 'audio_combined.wav'))
    else:
        shutil.copy(os.path.join(folder, 'audio_tts.wav'), os.path.join(folder, 'audio_combined.wav'))

    return f'Xử lý xong thư mục {folder}', os.path.join(folder, 'audio_combined.wav'), None

def init_TTS(method='edge'):
    """Khởi tạo môi trường cho Engine TTS."""
    from .factory import TTSFactory
    engine = TTSFactory.get_tts_engine(method)
    if hasattr(engine, '_init_model'):
        engine._init_model()
    elif hasattr(engine, '_init_env'):
        engine._init_env()


