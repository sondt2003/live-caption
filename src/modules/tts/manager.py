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
from .cn_tx import TextNorm

normalizer = TextNorm()

def preprocess_text(text):
    text = text.replace('AI', '人工智能')
    text = re.sub(r'(?<!^)([A-Z])', r' \1', text)
    text = normalizer(text)
    text = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)
    return text

def adjust_audio_length(wav_path, desired_length, sample_rate=24000, min_speed_factor=0.6, max_speed_factor=1.1):
    try:
        wav, sample_rate = librosa.load(wav_path, sr=sample_rate)
    except Exception as e:
        logger.warning(f"Audio load failed, returning silence: {e}")
        return np.zeros((int(desired_length * sample_rate), )), desired_length
    
    current_length = len(wav)/sample_rate
    speed_factor = max(min(desired_length / current_length, max_speed_factor), min_speed_factor)
    
    target_path = wav_path.replace('.wav', '_adjusted.wav').replace('.mp3', '_adjusted.wav')
    stretch_audio(wav_path, target_path, ratio=speed_factor, sample_rate=sample_rate)
    wav, sample_rate = librosa.load(target_path, sr=sample_rate)
    return wav[:int(desired_length*sample_rate)], desired_length

def generate_all_wavs_under_folder(folder, method, target_language='vi', voice='vi-VN-HoaiMyNeural'):
    transcript_path = os.path.join(folder, 'translation.json')
    output_folder = os.path.join(folder, 'wavs')
    os.makedirs(output_folder, exist_ok=True)

    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    # Normalize language code
    target_language = target_language.lower().replace('tiếng việt', 'vi').replace('简体中文', 'zh-cn')
    
    engine = TTSFactory.get_tts_engine(method)
    
    full_wav = np.zeros((0, ))
    for i, line in enumerate(transcript):
        speaker = line['speaker']
        text = preprocess_text(line['translation'])
        output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        
        # Speaker reference logic
        vocal_dereverb_wav = os.path.join(folder, 'audio_vocals_dereverb.wav')
        speaker_wav = vocal_dereverb_wav if os.path.exists(vocal_dereverb_wav) else os.path.join(folder, 'SPEAKER', f'{speaker}.wav')
        if not os.path.exists(speaker_wav): speaker_wav = os.path.join(folder, 'audio_vocals.wav')

        # Generate TTS
        engine.generate(text, output_path, speaker_wav=speaker_wav, prompt_text=line['text'], target_language=target_language, voice=voice)

        # Timeline alignment
        start, end = line['start'], line['end']
        length = end - start
        last_end = len(full_wav)/24000
        
        if start > last_end:
            full_wav = np.concatenate((full_wav, np.zeros((int((start - last_end) * 24000), ))))
        
        current_start = len(full_wav)/24000
        line['start'] = current_start
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            wav, adjusted_len = adjust_audio_length(output_path, length)
        else:
            wav = np.zeros((int(length * 24000), ))
            adjusted_len = length

        full_wav = np.concatenate((full_wav, wav))
        line['end'] = current_start + adjusted_len

    # Audio Mastering & Combining
    logger.info("Applying Studio-Grade Mastering...")
    try:
        meter = pyln.Meter(24000)
        loudness = meter.integrated_loudness(full_wav)
        full_wav = pyln.normalize.loudness(full_wav, loudness, -23.0)
    except Exception as e:
        logger.warning(f"Mastering failed: {e}")

    save_wav(full_wav, os.path.join(folder, 'audio_tts.wav'))
    
    # Combine with instruments if available
    instr_path = os.path.join(folder, 'audio_instruments.wav')
    if os.path.exists(instr_path):
        instr, _ = librosa.load(instr_path, sr=24000)
        # Ensure signals have the same length for addition
        min_len = min(len(full_wav), len(instr))
        combined = full_wav[:min_len] + instr[:min_len]
        save_wav_norm(combined, os.path.join(folder, 'audio_combined.wav'))
    else:
        shutil.copy(os.path.join(folder, 'audio_tts.wav'), os.path.join(folder, 'audio_combined.wav'))

    return f'Processed {folder}', os.path.join(folder, 'audio_combined.wav'), None

def init_TTS():
    from .factory import TTSFactory
    engine = TTSFactory.get_tts_engine('xtts')
    engine._init_model()

def init_cosyvoice():
    from .factory import TTSFactory
    engine = TTSFactory.get_tts_engine('cosyvoice')
    engine._init_model()


