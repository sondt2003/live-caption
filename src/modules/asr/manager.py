
import os
import torch
import numpy as np
from dotenv import load_dotenv
from .whisperx import whisperx_transcribe_audio, load_align_model
from .google_speech import google_transcribe_audio
from utils.utils import save_wav
import json
import librosa
from loguru import logger
load_dotenv()

def merge_segments(transcript, ending='!"\').:;?]}~！“”’）。：；？】'):
    merged_transcription = []
    buffer_segment = None

    for segment in transcript:
        if buffer_segment is None:
            buffer_segment = segment
        else:
            # Check if the last character of the 'text' field is a punctuation mark
            if buffer_segment['text'][-1] in ending:
                # If it is, add the buffered segment to the merged transcription
                merged_transcription.append(buffer_segment)
                buffer_segment = segment
            else:
                # If it's not, merge this segment with the buffered segment
                buffer_segment['text'] += ' ' + segment['text']
                buffer_segment['end'] = segment['end']

    # Don't forget to add the last buffered segment
    if buffer_segment is not None:
        merged_transcription.append(buffer_segment)

    return merged_transcription

def generate_speaker_audio(folder, transcript, audio_data=None):
    if audio_data is None:
        wav_path = os.path.join(folder, 'audio_vocals.wav')
        if not os.path.exists(wav_path):
            logger.warning(f"Vocals file not found for speaker audio generation: {wav_path}")
            return
        audio_data, samplerate = librosa.load(wav_path, sr=24000)
    else:
        samplerate = 24000 # WhisperX load_audio returns 16k by default, but our factory logic might expect consistent SR. 
        # Actually, whisperx.load_audio returns 16000. Let's check whisperx.py
        samplerate = 16000 # Standard for WhisperX
        
    speaker_dict = dict()
    length = len(audio_data)
    delay = 0.05
    for segment in transcript:
        start = max(0, int((segment['start'] - delay) * samplerate))
        end = min(int((segment['end']+delay) * samplerate), length)
        speaker_segment_audio = audio_data[start:end]
        speaker_dict[segment['speaker']] = np.concatenate((speaker_dict.get(
            segment['speaker'], np.zeros((0, ))), speaker_segment_audio))

    speaker_folder = os.path.join(folder, 'SPEAKER')
    if not os.path.exists(speaker_folder):
        os.makedirs(speaker_folder)
    
    for speaker, audio in speaker_dict.items():
        speaker_file_path = os.path.join(
            speaker_folder, f"{speaker}.wav")
        save_wav(audio, speaker_file_path, sample_rate=samplerate)


def transcribe_audio(folder, model_name: str = 'large', download_root='models/ASR/whisper', device='auto', batch_size=32, diarization=True, min_speakers=None, max_speakers=None, language=None, asr_method='whisperx', google_key=None):
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    
    if not os.path.exists(wav_path):
        logger.error(f"Vocals file not found for transcription: {wav_path}")
        return False
    
    logger.info(f'Transcribing {wav_path} using {asr_method}')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if asr_method == 'google' and google_key:
        # 1. Get raw text from Google
        raw_transcript = google_transcribe_audio(wav_path, google_key, lang=language or 'zh')
        
        # 2. Hybrid Alignment: Use local WhisperX alignment for precision
        import whisperx as wx
        audio = wx.load_audio(wav_path)
        
        # Convert Google format to WhisperX alignment compatible format
        # WhisperX align expects a list of segment dicts with 'text', 'start', 'end'
        segments = [{"text": t['text'], "start": t['start'], "end": t['end']} for t in raw_transcript]
        
        # Detect language if not provided (fallback)
        detect_lang = language or 'zh' 
        
        # Load local alignment model
        from .whisperx import load_align_model
        model_dir = download_root
        align_model, align_metadata = wx.load_align_model(language_code=detect_lang, device=device, model_dir=model_dir)
        
        if align_model:
            logger.info(f"Hybrid Alignment: Aligning Google transcript with local models...")
            result = wx.align(segments, align_model, align_metadata, audio, device, return_char_alignments=False)
            transcript = []
            for segment in result['segments']:
                transcript.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip(),
                    'speaker': 'SPEAKER_00' # Google V2 doesn't do diarization
                })
        else:
            # Fallback to Google's approximated timestamps if alignment fails
            transcript = raw_transcript
        
        audio_data = audio
    else:
        # Default: WhisperX
        transcript, audio_data = whisperx_transcribe_audio(wav_path, model_name, download_root, device, batch_size, diarization, min_speakers, max_speakers, language=language)

    if not transcript:
        raise Exception("Không tìm thấy nội dung giọng nói nào (nhận diện thất bại hoặc audio không có tiếng).")
    
    with open(os.path.join(folder, 'transcript.json'), 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=4, ensure_ascii=False)
    logger.info(f'Transcribed {wav_path} successfully, and saved to {os.path.join(folder, "transcript.json")}')
    generate_speaker_audio(folder, transcript, audio_data=audio_data)
    return transcript

def transcribe_all_audio_under_folder(folder, whisper_model_name: str = 'large', device='auto', batch_size=32, diarization=False, min_speakers=None, max_speakers=None, language=None, asr_method='whisperx', google_key=None):
    transcribe_json = None
    for root, dirs, files in os.walk(folder):
        if 'audio_vocals.wav' in files and 'transcript.json' not in files:
            transcribe_json = transcribe_audio(root, whisper_model_name, 'models/ASR/whisper', device, batch_size, diarization, min_speakers, max_speakers, language=language, asr_method=asr_method, google_key=google_key)
        elif 'transcript.json' in files:
            transcribe_json = json.load(open(os.path.join(root, 'transcript.json'), 'r', encoding='utf-8'))

            # logger.info(f'Transcript already exists in {root}')
    return f'Transcribed all audio under {folder}', transcribe_json

if __name__ == '__main__':
    _, transcribe_json = transcribe_all_audio_under_folder('videos', 'WhisperX')
    print(transcribe_json)
    # _, transcribe_json = transcribe_all_audio_under_folder('videos', 'FunASR')    
    # print(transcribe_json)