import os
import torch
import gc
import json
import librosa
import numpy as np
from loguru import logger
from dotenv import load_dotenv
from src.modules.translation.manager import split_text_into_sentences
from .whisperx import whisperx_transcribe_audio, load_align_model
from .google_speech import google_transcribe_audio
from src.utils.utils import save_wav

load_dotenv()

def merge_segments(transcript, ending='!"\').:;?]}~！“”’）。：；？】'):
    merged = []
    buf = None
    for seg in transcript:
        if buf is None: buf = seg
        else:
            if buf['text'][-1] in ending:
                merged.append(buf)
                buf = seg
            else:
                buf['text'] += ' ' + seg['text']
                buf['end'] = seg['end']
    if buf: merged.append(buf)
    return merged

def generate_speaker_audio(folder, transcript, audio_data=None):
    if audio_data is None:
        wav_path = os.path.join(folder, 'audio_vocals.wav')
        if not os.path.exists(wav_path): return
        audio_data, sr = librosa.load(wav_path, sr=24000)
    else: sr = 16000
    spk_dict = {}
    length = len(audio_data)
    for seg in transcript:
        start = max(0, int((seg['start'] - 0.05) * sr))
        end = min(int((seg['end'] + 0.05) * sr), length)
        spk_dict[seg['speaker']] = np.concatenate((spk_dict.get(seg['speaker'], np.zeros((0,))), audio_data[start:end]))
    spk_folder = os.path.join(folder, 'SPEAKER')
    if not os.path.exists(spk_folder): os.makedirs(spk_folder)
    for s, a in spk_dict.items():
        save_wav(a, os.path.join(spk_folder, f"{s}.wav"), sample_rate=sr)

def transcribe_audio(folder, model_name='large', download_root='models/ASR/whisper', device='auto', batch_size=32, diarization=True, min_speakers=None, max_speakers=None, language=None, asr_method='whisperx', google_key=None):
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    if not os.path.exists(wav_path): return False
    if device == 'auto': device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Global SpeechBrain LID
    if not language:
        logger.info("Auto-Language Detection using SpeechBrain...")
        try:
            from speechbrain.inference import EncoderClassifier
            classifier = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="models/LID")
            wav_path_lid = wav_path
            signal = classifier.load_audio(wav_path_lid)
            if signal.shape[0] > 16000*30: signal = signal[:16000*30]
            prediction = classifier.classify_batch(signal.unsqueeze(0))
            lang_label = prediction[3][0]
            language = lang_label.split(":")[0]
            logger.info(f"SpeechBrain Detected Language: {language}")
            del classifier
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"SpeechBrain LID failed: {e}. Defaulting to 'zh'.")
            language = 'zh'
    
    if asr_method == 'google':
        logger.info(f"Using Google ASR (Free/Web API) for language: {language}")
        raw = google_transcribe_audio(wav_path, google_key, lang=language)
        import whisperx as wx
        audio = wx.load_audio(wav_path)
        segs = [{"text": t['text'], "start": t['start'], "end": t['end']} for t in raw]
        from .whisperx import load_align_model
        aln, meta = wx.load_align_model(language_code=language, device=device, model_dir=download_root)
        if aln:
            res = wx.align(segs, aln, meta, audio, device, return_char_alignments=False)
            transcript = [{'start': s['start'], 'end': s['end'], 'text': s['text'].strip(), 'speaker': 'SPEAKER_00'} for s in res['segments']]
        else: transcript = raw
        audio_data = audio
    else:
        logger.info(f"Using WhisperX ASR for language: {language}")
        transcript, audio_data = whisperx_transcribe_audio(wav_path, model_name, download_root, device, batch_size, diarization, min_speakers, max_speakers, language=language)

    if not transcript: raise Exception("No speech found")

    new_t = []
    for s in transcript:
        sents = split_text_into_sentences(s['text'])
        if len(sents) > 1:
            total = len(s['text'])
            cur = s['start']
            dur = s['end'] - s['start']
            for st in sents:
                if not st.strip(): continue
                sdur = (len(st)/total)*dur
                new_t.append({'start':round(cur,3), 'end':round(cur+sdur,3), 'text':st.strip(), 'speaker':s.get('speaker','SPEAKER_00')})
                cur += sdur
        else: new_t.append(s)
    transcript = new_t

    with open(os.path.join(folder, 'transcript.json'), 'w', encoding='utf-8') as f: json.dump(transcript, f, indent=4, ensure_ascii=False)
    generate_speaker_audio(folder, transcript, audio_data)
    return transcript

def transcribe_all_audio_under_folder(folder, whisper_model_name='large', device='auto', batch_size=32, diarization=False, min_speakers=None, max_speakers=None, language=None, asr_method='whisperx', google_key=None):
    t_json = None
    for root, dirs, files in os.walk(folder):
        if 'audio_vocals.wav' in files and 'transcript.json' not in files:
            t_json = transcribe_audio(root, whisper_model_name, 'models/ASR/whisper', device, batch_size, diarization, min_speakers, max_speakers, language=language, asr_method=asr_method, google_key=google_key)
        elif 'transcript.json' in files:
            t_json = json.load(open(os.path.join(root, 'transcript.json'), 'r', encoding='utf-8'))
    return 'Done', t_json
