import json
import os
import re
import librosa
import shutil
import numpy as np
import pyloudnorm as pyln
from loguru import logger
import subprocess
from src.utils.utils import save_wav, save_wav_norm
from .factory import TTSFactory

def stretch_audio_ffmpeg(input_path, output_path, rate, sample_rate=24000):
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
    cmd = ['ffmpeg', '-i', input_path, '-filter:a', filter_str, '-ar', str(sample_rate), '-ac', '1', output_path, '-y']
    subprocess.run(cmd, capture_output=True)

def preprocess_text(text, target_language='vi'):
    if 'zh' in target_language.lower():
        text = text.replace('AI', '人工智能')
        text = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)
    else:
        text = text.replace('AI', 'A I')
        text = re.sub(r'\s+', ' ', text).strip()
    return text

class VoiceMapper:
    def __init__(self, target_language='vi', default_voice='vi-VN-HoaiMyNeural'):
        self.target_language = target_language.lower()
        self.default_voice = default_voice
        self.mapping = self._load_mapping()
        self.speaker_cache = {}
        self.pools = {
            'vi': {'male': 'vi-VN-NamMinhNeural', 'female': 'vi-VN-HoaiMyNeural'},
            'zh-cn': {'male': 'zh-CN-YunxiNeural', 'female': 'zh-CN-XiaoxiaoNeural'},
            'en': {'male': 'en-US-GuyNeural', 'female': 'en-US-AriaNeural'}
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
        if speaker_id in self.mapping: return self.mapping[speaker_id]
        if speaker_id in self.speaker_cache: return self.speaker_cache[speaker_id]
        lang_key = 'vi' if 'vi' in self.target_language else ('zh-cn' if 'zh' in self.target_language else 'en')
        pool = self.pools.get(lang_key, self.pools['en'])
        if lang_key == 'vi':
            male_keywords = ['anh', 'ông', 'chú', 'bác', 'cậu', 'ngài', 'nam']
            female_keywords = ['chị', 'bà', 'cô', 'dì', 'mợ', 'nữ']
            text_lower = text.lower()
            m_score = sum(1 for k in male_keywords if re.search(fr'\b{k}\b', text_lower))
            f_score = sum(1 for k in female_keywords if re.search(fr'\b{k}\b', text_lower))
            if m_score > f_score:
                self.speaker_cache[speaker_id] = pool['male']
                return pool['male']
            elif f_score > m_score:
                self.speaker_cache[speaker_id] = pool['female']
                return pool['female']
        try:
            nums = re.findall(r'\d+', speaker_id)
            idx = int(nums[0]) if nums else len(self.speaker_cache)
        except: idx = len(self.speaker_cache)
        voice = pool['male'] if idx % 2 == 1 else self.default_voice
        self.speaker_cache[speaker_id] = voice
        return voice

def adjust_audio_length(wav_path, desired_length, sample_rate=44100, min_speed_factor=0.01, max_speed_factor=10.0):
    try:
        wav_orig, _ = librosa.load(wav_path, sr=sample_rate)
        current_length = len(wav_orig) / sample_rate
    except Exception as e:
        return np.zeros((int(desired_length * sample_rate), )), desired_length
    if current_length < desired_length:
        padding = np.zeros((int((desired_length - current_length) * sample_rate), ))
        return np.concatenate((wav_orig, padding)), desired_length
    ratio = desired_length / current_length
    final_ratio = max(ratio, min_speed_factor)
    try:
        rate = 1.0 / final_ratio
        target_path = wav_path.replace('.wav', '_stretched.wav')
        stretch_audio_ffmpeg(wav_path, target_path, rate, sample_rate=sample_rate)
        if os.path.exists(target_path): wav_final, _ = librosa.load(target_path, sr=sample_rate)
        else: raise Exception("FFmpeg failed")
    except: wav_final = wav_orig[:int(desired_length * sample_rate)]
    return wav_final, len(wav_final) / sample_rate

def get_gender_from_audio(wav_path, start, end, audio_data=None, sr=16000):
    try:
        duration = end - start
        if duration < 0.5: return None
        if audio_data is not None:
            start_s = int(start * sr)
            end_s = int(end * sr)
            y = audio_data[start_s:end_s]
        else: y, sr = librosa.load(wav_path, sr=sr, offset=start, duration=duration)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=300, sr=sr)
        voiced_f0 = f0[voiced_flag]
        return np.mean(voiced_f0) if len(voiced_f0) > 0 else None
    except: return None

def generate_all_wavs_under_folder(folder, method='auto', target_language='vi', voice='vi-VN-HoaiMyNeural', video_volume=1.0):
    transcript_path = os.path.join(folder, 'translation.json')
    output_folder = os.path.join(folder, 'wavs')
    os.makedirs(output_folder, exist_ok=True)
    with open(transcript_path, 'r', encoding='utf-8') as f: transcript = json.load(f)

    target_language = target_language.lower()
    if 'vi' in target_language: target_language = 'vi'
    elif 'zh' in target_language: target_language = 'zh-cn'
    elif 'en' in target_language: target_language = 'en'
    
    speaker_gender_map = {}
    if 'vi' in target_language:
        pitches = {}
        unique_spk = set(l['speaker'] for l in transcript)
        force_male = len(unique_spk) <= 1
        vocals_path = os.path.join(folder, 'audio_vocals.wav')
        audio_v = None
        sr_v = 44100
        if os.path.exists(vocals_path): audio_v, _ = librosa.load(vocals_path, sr=sr_v)
        for line in transcript:
            spk = line['speaker']
            if force_male: speaker_gender_map[spk] = 'male'; continue
            if spk not in pitches: pitches[spk] = []
            if len(pitches[spk]) < 5:
                p = get_gender_from_audio(vocals_path, line.get('start'), line.get('end'), audio_data=audio_v, sr=sr_v)
                if p: pitches[spk].append(p)
        for spk, p_list in pitches.items():
            if not p_list: speaker_gender_map[spk] = 'female'
            else: speaker_gender_map[spk] = 'male' if np.mean(p_list) < 165 else 'female'

    engine = TTSFactory.get_best_tts_engine(target_language) if method in [None, 'auto'] else TTSFactory.get_tts_engine(method)
    voice_mapper = VoiceMapper(target_language=target_language, default_voice=voice)
    tasks = []
    for i, line in enumerate(transcript):
        speaker = line['speaker']
        text = preprocess_text(line['translation'], target_language)
        out_p = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        spk_wav = os.path.join(folder, 'SPEAKER', f'{speaker}.wav')
        if not os.path.exists(spk_wav): spk_wav = os.path.join(folder, 'audio_vocals.wav')
        t_voice = voice_mapper.get_voice(speaker, text)
        if (voice in [None, 'auto', 'vi-VN-HoaiMyNeural']) and 'vi' in target_language and "EdgeTTS" in str(type(engine)):
            t_voice = 'vi-VN-NamMinhNeural' if speaker_gender_map.get(speaker, 'female') == 'male' else 'vi-VN-HoaiMyNeural'
        if not text.strip():
            logger.warning(f"Skipping empty text for segment {i}")
            continue
        tasks.append({"text": text, "output_path": out_p, "speaker_wav": spk_wav, "ref_text": None, "target_language": target_language, "voice": t_voice})

    if hasattr(engine, 'generate_batch'): engine.generate_batch(tasks)
    else:
        for t in tasks: engine.generate(t.pop("text"), t.pop("output_path"), **t)

    full_wav = np.zeros((0, ))
    TARGET_SR = 44100
    current_out_end = 0.0
    MIN_GAP = float(os.getenv('MIN_GAP', 0))
    MAX_PTS_FACTOR = float(os.getenv('MAX_PTS_FACTOR', 1.0))
    
    for i, line in enumerate(transcript):
        if 'original_start' not in line: line['original_start'] = line.get('start', 0.0)
        if 'original_end' not in line: line['original_end'] = line.get('end', 0.0)
        output_p = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        t_start = max(line['original_start'], current_out_end + MIN_GAP)
        if t_start > current_out_end:
            full_wav = np.concatenate((full_wav, np.zeros((int((t_start - current_out_end) * TARGET_SR), ))))
        c_start = len(full_wav)/TARGET_SR
        line['start'] = c_start
        if os.path.exists(output_p) and os.path.getsize(output_p) > 0:
            raw_vox, _ = librosa.load(output_p, sr=TARGET_SR)
            raw_d = len(raw_vox) / TARGET_SR
            o_dur = line['original_end'] - line['original_start']
            max_s = o_dur * MAX_PTS_FACTOR
            max_e = line['original_end'] * MAX_PTS_FACTOR
            ideal_d = max(0.2, line['original_end'] - t_start)
            hard_d = max(0.2, min(max_s, max_e - t_start))
            stretch_t = ideal_d if (ideal_d/raw_d if raw_d > 0 else 1.0) >= 0.5 else hard_d
            wav, adj_l = adjust_audio_length(output_p, stretch_t, min_speed_factor=0.15)
        else:
            wav = np.zeros((int((line['original_end']-line['original_start']) * TARGET_SR), ))
            adj_l = line['original_end'] - line['original_start']
        full_wav = np.concatenate((full_wav, wav))
        line['end'] = c_start + adj_l
        current_out_end = line['end']
        line['duration'] = adj_l

    with open(transcript_path, 'w', encoding='utf-8') as f: json.dump(transcript, f, ensure_ascii=False, indent=2)
    try:
        meter = pyln.Meter(TARGET_SR)
        loudness = meter.integrated_loudness(full_wav)
        if not np.isinf(loudness): full_wav = pyln.normalize.loudness(full_wav, loudness, -23.0)
    except: pass

    try:
        instr_path = os.path.join(folder, 'audio_instruments.wav')
        orig_t_dur = 0
        if os.path.exists(instr_path): orig_t_dur = librosa.get_duration(path=instr_path)
        elif os.path.exists(os.path.join(folder, 'audio_vocals.wav')):
            orig_t_dur = librosa.get_duration(path=os.path.join(folder, 'audio_vocals.wav'))
            
        if orig_t_dur > 0 and transcript:
            final_v_dur = transcript[-1]['end'] + max(0, orig_t_dur - transcript[-1]['original_end'])
            curr_a_dur = len(full_wav) / TARGET_SR
            if final_v_dur > curr_a_dur:
                full_wav = np.concatenate([full_wav, np.zeros(int((final_v_dur - curr_a_dur) * TARGET_SR))])
    except: pass

    save_wav_norm(full_wav, os.path.join(folder, 'audio_tts.wav'))
    try: shutil.rmtree(output_folder)
    except: pass
    
    if os.path.exists(instr_path):
        instr, _ = librosa.load(instr_path, sr=TARGET_SR, mono=False)
        if instr.ndim > 1:
            if instr.shape[1] < len(full_wav):
                instr = np.concatenate([instr, np.zeros((2, len(full_wav) - instr.shape[1]))], axis=1)
            f_wav_s = np.tile(full_wav, (2, 1))
            combined = f_wav_s + instr[:, :len(full_wav)] * video_volume
        else:
            if len(instr) < len(full_wav):
                instr = np.concatenate([instr, np.zeros(len(full_wav) - len(instr))])
            combined = full_wav + instr[:len(full_wav)] * video_volume
        save_wav_norm(combined, os.path.join(folder, 'audio_combined.wav'), sample_rate=TARGET_SR)
    else:
        save_wav_norm(full_wav, os.path.join(folder, 'audio_combined.wav'), sample_rate=TARGET_SR)
    return f'Done {folder}', os.path.join(folder, 'audio_combined.wav'), None

def init_TTS(method='edge'):
    from .factory import TTSFactory
    engine = TTSFactory.get_tts_engine(method)
    if hasattr(engine, '_init_model'): engine._init_model()
    elif hasattr(engine, '_init_env'): engine._init_env()
