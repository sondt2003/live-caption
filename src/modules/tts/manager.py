import json
import os
import re
import librosa
import shutil
import numpy as np
import pyloudnorm as pyln
from loguru import logger
import subprocess
from utils.utils import save_wav, save_wav_norm
from .factory import TTSFactory

def stretch_audio_ffmpeg(input_path, output_path, rate, sample_rate=24000):
    """Sử dụng FFmpeg atempo để thay đổi tốc độ âm thanh với chất lượng cao hơn librosa."""
    # atempo chỉ hỗ trợ 0.5 đến 2.0. Phải nối chuỗi nếu ngoài khoảng này.
    filters = []
    temp_rate = rate
    while temp_rate > 2.0:
        filters.append("atempo=2.0")
        temp_rate /= 2.0
    while temp_rate < 0.5:
        filters.append("atempo=0.5")
        temp_rate /= 0.5
    filters.append(f"atempo={temp_rate:.4f}")
    
    filter_str = ",".join(filters)
    cmd = [
        'ffmpeg', '-i', input_path,
        '-filter:a', filter_str,
        '-ar', str(sample_rate),
        '-ac', '1',
        output_path, '-y'
    ]
    subprocess.run(cmd, capture_output=True)

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

class VoiceMapper:
    def __init__(self, target_language='vi', default_voice='vi-VN-HoaiMyNeural'):
        self.target_language = target_language.lower()
        self.default_voice = default_voice
        self.mapping = self._load_mapping()
        self.speaker_cache = {}
        
        # Default pools (mostly for Edge TTS)
        self.pools = {
            'vi': {
                'male': 'vi-VN-NamMinhNeural',
                'female': 'vi-VN-HoaiMyNeural'
            },
            'zh-cn': {
                'male': 'zh-CN-YunxiNeural',
                'female': 'zh-CN-XiaoxiaoNeural'
            },
            'en': {
                'male': 'en-US-GuyNeural',
                'female': 'en-US-AriaNeural'
            }
        }
        
    def _load_mapping(self):
        mapping_str = os.getenv("VOICE_MAPPING", "")
        mapping = {}
        if mapping_str:
            for item in mapping_str.split(","):
                if ":" in item:
                    k, v = item.split(":", 1)
                    mapping[k.strip()] = v.strip()
        return mapping

    def get_voice(self, speaker_id, text=""):
        # 1. Ưu tiên mapping cứng từ .env
        if speaker_id in self.mapping:
            return self.mapping[speaker_id]
        
        # 2. Nếu đã gán cho speaker này rồi thì dùng lại
        if speaker_id in self.speaker_cache:
            return self.speaker_cache[speaker_id]

        # 3. Phán đoán giới tính dựa trên đại từ (Tiếng Việt)
        lang_key = 'vi' if 'vi' in self.target_language else ('zh-cn' if 'zh' in self.target_language else 'en')
        pool = self.pools.get(lang_key, self.pools['en'])

        if lang_key == 'vi':
            male_keywords = ['anh', 'ông', 'chú', 'bác', 'cậu', 'ngài', 'nam']
            female_keywords = ['chị', 'bà', 'cô', 'dì', 'mợ', 'nữ']
            
            text_lower = text.lower()
            male_score = sum(1 for k in male_keywords if re.search(fr'\b{k}\b', text_lower))
            female_score = sum(1 for k in female_keywords if re.search(fr'\b{k}\b', text_lower))
            
            if male_score > female_score:
                self.speaker_cache[speaker_id] = pool['male']
                return pool['male']
            elif female_score > male_score:
                self.speaker_cache[speaker_id] = pool['female']
                return pool['female']

        # 4. Tự động xen kẽ nếu có nhiều speaker (SPEAKER_00, 01...)
        try:
            nums = re.findall(r'\d+', speaker_id)
            idx = int(nums[0]) if nums else len(self.speaker_cache)
        except:
            idx = len(self.speaker_cache)
            
        if idx % 2 == 1:
            # Nếu speaker lẻ, đổi sang giọng khác với mặc định
            voice = pool['male'] if self.default_voice == pool['female'] else pool['female']
        else:
            voice = self.default_voice
            
        self.speaker_cache[speaker_id] = voice
        return voice

def adjust_audio_length(wav_path, desired_length, sample_rate=24000, min_speed_factor=0.5, max_speed_factor=1.35):
    try:
        # Load directly to numpy for manipulation
        wav_orig, _ = librosa.load(wav_path, sr=sample_rate)
        current_length = len(wav_orig) / sample_rate
    except Exception as e:
        logger.warning(f"Audio load failed, returning silence: {e}")
        return np.zeros((int(desired_length * sample_rate), )), desired_length

    # CASE 1: Audio is shorter (Needs lengthening) -> Pad with silence
    if current_length < desired_length:
        gap = desired_length - current_length
        padding = np.zeros((int(gap * sample_rate), ))
        wav_final = np.concatenate((wav_orig, padding))
        return wav_final, desired_length

    # CASE 2: Audio is longer (Needs shortening) -> Stretch (speed up)
    ratio = desired_length / current_length
    final_ratio = max(ratio, min_speed_factor)
    
    try:
        # FFmpeg expects rate (1/final_ratio). rate > 1.0 speeds up.
        rate = 1.0 / final_ratio
        target_path = wav_path.replace('.wav', '_stretched.wav')
        stretch_audio_ffmpeg(wav_path, target_path, rate, sample_rate=sample_rate)
        
        if os.path.exists(target_path):
            wav_final, _ = librosa.load(target_path, sr=sample_rate)
        else:
            raise Exception("FFmpeg failed to generate stretched audio")
            
    except Exception as ex:
        logger.error(f"Time stretch failed: {ex}. Using original.")
        # Crop as fallback
        wav_final = wav_orig[:int(desired_length * sample_rate)]

    actual_duration = len(wav_final) / sample_rate
    return wav_final, actual_duration

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

    # 1. Initialize VoiceMapper
    voice_mapper = VoiceMapper(target_language=target_language, default_voice=voice)
    
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
            "voice": voice_mapper.get_voice(speaker, text)
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
        # Backup original timings (STABLE REFERENCE)
        # Chỉ backup nếu chưa có (tránh việc lặp lại làm hỏng timings gốc)
        if 'original_start' not in line:
            line['original_start'] = line.get('start', 0.0)
        if 'original_end' not in line:
            line['original_end'] = line.get('end', 0.0)
        
        output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        last_end = len(full_wav)/24000
        
    # Timeline Pacing Logic
    current_out_end = 0.0
    
    # Đọc cấu hình từ .env
    from dotenv import load_dotenv
    load_dotenv()
    
    # Khoảng lặng tối thiểu giữa các câu (giây).
    # Tăng lên để giọng đọc tự nhiên hơn, giảm xuống để video nhanh hơn.
    MIN_GAP = float(os.getenv('MIN_GAP', 0))
    
    # Giới hạn hệ số giãn nở video tối đa.
    # 1.43 nghĩa là video chỉ được phép chậm lại tối đa 43%.
    # Nếu quá giới hạn này, âm thanh sẽ bị tua nhanh thay vì video chậm thêm.
    MAX_PTS_FACTOR = float(os.getenv('MAX_PTS_FACTOR', 1.43))
    
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
            
            # PHYSICAL VIDEO LIMIT: A video segment cannot stretch more than 1.43x.
            # Thus, audio SHOULD NOT be longer than orig_dur * 1.43.
            max_seg_allowed = orig_dur * MAX_PTS_FACTOR
            
            # ABSOLUTE DEADLINE: Audio must finish before this point to keep entire video in sync.
            max_end_allowed = orig_end * MAX_PTS_FACTOR
            
            # 1. Ideal Target: Finish at original English time (Real-time sync)
            ideal_dur = max(0.2, orig_end - target_start)
            
            # 2. Hard Limit: Cannot exceed physical video stretch
            hard_limit_dur = max(0.2, min(max_seg_allowed, max_end_allowed - target_start))
            
            # 3. Decision: Use ideal if it doesn't require extreme compression, else use hard limit
            needed_ratio_to_ideal = ideal_dur / raw_dur if raw_dur > 0 else 1.0
            
            if needed_ratio_to_ideal >= 0.6: # If < 1.6x speedup is enough to catch up
                stretch_to = ideal_dur
                logger.debug(f"Segment {i}: Catch-up target (Ideal: {ideal_dur:.2f}s)")
            else:
                # Catching up is too hard, at least try to stay within the 1.43x video limit
                stretch_to = hard_limit_dur
                logger.debug(f"Segment {i}: Video-limit target (Max: {hard_limit_dur:.2f}s)")

            # We allow compression up to 4x (0.25) to fit these strict limits
            local_min_speed_factor = 0.25
            
            wav, adjusted_len = adjust_audio_length(output_path, stretch_to, min_speed_factor=local_min_speed_factor)
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

    # ĐỒNG BỘ ĐỘ DÀI: Đảm bảo âm thanh dài bằng video (bao gồm cả đoạn 'tail')
    try:
        # Tìm metadata video để lấy thời lượng gốc
        # Chúng ta giả định audio_instruments.wav hoặc audio_vocals.wav có thời lượng bằng video gốc
        orig_audio_path = os.path.join(folder, 'audio_instruments.wav')
        if not os.path.exists(orig_audio_path):
            orig_audio_path = os.path.join(folder, 'audio_vocals.wav')
            
        if os.path.exists(orig_audio_path):
            orig_total_dur = librosa.get_duration(path=orig_audio_path)
            # Theo logic video.py: final_v_dur = last_target_end + (orig_total_dur - last_orig_end)
            if transcript:
                last_line = transcript[-1]
                last_orig_end = last_line['original_end']
                last_target_end = last_line['end']
                
                final_v_dur = last_target_end + max(0, orig_total_dur - last_orig_end)
                current_audio_dur = len(full_wav) / 24000
                
                if final_v_dur > current_audio_dur:
                    padding_len = int((final_v_dur - current_audio_dur) * 24000)
                    full_wav = np.concatenate([full_wav, np.zeros(padding_len)])
                    logger.info(f"Đã thêm {final_v_dur - current_audio_dur:.2f}s khoảng lặng vào cuối để khớp với video.")

    except Exception as e:
        logger.warning(f"Không thể đồng bộ độ dài audio với video: {e}")

    save_wav_norm(full_wav, os.path.join(folder, 'audio_tts.wav'))
    
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


