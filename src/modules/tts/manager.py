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

def adjust_audio_length(wav_path, desired_length, sample_rate=24000, min_speed_factor=0.5, max_speed_factor=1.35):
    try:
        # Load to check length
        wav_orig, _ = librosa.load(wav_path, sr=sample_rate)
        current_length = len(wav_orig)/sample_rate
    except Exception as e:
        logger.warning(f"Audio load failed, returning silence: {e}")
        return np.zeros((int(desired_length * sample_rate), )), desired_length
    
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
        
    # Timeline Pacing Logic
    current_out_end = 0.0
    MIN_GAP = 0.1 # Minimum gap between segments in the output
    MAX_PTS_FACTOR = 1.43 # Giới hạn giãn video (Must be <= 1.5 to stay natural)
    
    for i, line in enumerate(transcript):
        output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        orig_start = line['original_start']
        orig_end = line['original_end']
        orig_dur = orig_end - orig_start
        
        # 1. Determine the earliest possible start time
        # We try to start at the original time, but must follow previous output
        target_start = max(orig_start, current_out_end + MIN_GAP)
        
        # 2. Add silence to reach the target start in the main wav
        if target_start > current_out_end:
            full_wav = np.concatenate((full_wav, np.zeros((int((target_start - current_out_end) * 24000), ))))
        
        current_start = len(full_wav)/24000
        line['start'] = current_start
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            # Load raw duration for calculation
            raw_vox, _ = librosa.load(output_path, sr=24000)
            raw_dur = len(raw_vox) / 24000
            
            # CƠ CHẾ ĐỒNG BỘ CỨNG:
            # Không được để đoạn audio dài hơn độ giãn video tối đa (1.43x)
            max_dur_allowed = orig_dur * MAX_PTS_FACTOR
            
            # Nếu bản dịch quá dài, ta phải ép nó ngắn lại cho vừa max_dur_allowed
            # Nếu bản dịch ngắn hơn, ta cố gắng khớp vào orig_dur hoặc available_dur
            stretch_to_raw = min(max_dur_allowed, max(orig_dur, raw_dur))
            
            # Nếu chúng ta đang bị chậm (target_start > orig_start), 
            # chúng ta có ít "không gian" hơn để giãn video.
            available_dur = max(0.5, (orig_end * MAX_PTS_FACTOR) - target_start)
            stretch_to = min(stretch_to_raw, available_dur)
            
            # Thực hiện co giãn audio
            wav, adjusted_len = adjust_audio_length(output_path, stretch_to)
            line['source_duration'] = orig_dur
        else:
            logger.warning(f"Segment {i}: Output path {output_path} not found. Filling with silence.")
            line['source_duration'] = orig_dur
            wav = np.zeros((int(orig_dur * 24000), ))
            adjusted_len = orig_dur

        full_wav = np.concatenate((full_wav, wav))
        line['end'] = current_start + adjusted_len
        current_out_end = line['end']
        line['duration'] = adjusted_len

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
    
    # Dọn dẹp các tệp tạm để tiết kiệm không gian
    try:
        shutil.rmtree(output_folder)
        logger.info(f"Đã dọn dẹp thư mục tệp tạm TTS: {output_folder}")
    except Exception as e:
        logger.warning(f"Không thể dọn dẹp thư mục tệp tạm TTS: {e}")
    
    # Trộn với nhạc nền/âm thanh gốc nếu có
    instr_path = os.path.join(folder, 'audio_instruments.wav')
    if os.path.exists(instr_path):
        instr, _ = librosa.load(instr_path, sr=24000)
        
        # ĐỒNG BỘ ĐỘ DÀI: Đảm bảo nhạc nền dài bằng âm thanh TTS (padding silence nếu cần)
        if len(instr) < len(full_wav):
            padding = np.zeros(len(full_wav) - len(instr))
            instr = np.concatenate([instr, padding])
        
        # Áp dụng video_volume cho nhạc nền, full_wav là giọng nói (giữ nguyên độ lớn)
        combined = full_wav + instr[:len(full_wav)] * video_volume
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


