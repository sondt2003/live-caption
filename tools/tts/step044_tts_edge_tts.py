import os
from loguru import logger
import numpy as np
import torch
import time
from ..utils.utils import save_wav
import sys
import subprocess

import torchaudio
model = None



#  <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
language_map = {
    '中文': 'zh-CN-XiaoxiaoNeural',
    'English': 'en-US-MichelleNeural',
    'Japanese': 'ja-JP-NanamiNeural',
    '粤语': 'zh-HK-HiuMaanNeural',
    'Korean': 'ko-KR-SunHiNeural',
    'Vietnamese': 'vi-VN-HoaiMyNeural',
    'Tiếng Việt': 'vi-VN-HoaiMyNeural',
}

def tts(text, output_path, target_language='中文', voice = 'zh-CN-XiaoxiaoNeural'):
    if os.path.exists(output_path):
        logger.info(f'TTS {text} 已存在')
        return
    
    # Get the path to edge-tts in the virtual environment
    edge_tts_path = os.path.join(os.path.dirname(sys.executable), 'edge-tts')
    
    mp3_path = output_path.replace(".wav", ".mp3")
    for retry in range(3):
        try:
            # Generate MP3 using EdgeTTS with full path
            result = subprocess.run([edge_tts_path, '--text', text, '--write-media', mp3_path, '--voice', voice], 
                                    capture_output=True, text=True)
            
            # Convert MP3 to WAV
            if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
                audio, sr = torchaudio.load(mp3_path)
                torchaudio.save(output_path, audio, sr)
                os.remove(mp3_path)  # Clean up MP3 file
                logger.info(f'TTS {text}')
                break
            else:
                logger.warning(f'TTS {text} failed - MP3 file not created or empty')
                if result.stderr:
                    logger.warning(f'EdgeTTS error: {result.stderr}')
        except Exception as e:
            logger.warning(f'TTS {text} 失败')
            logger.warning(e)


if __name__ == '__main__':
    speaker_wav = r'videos/村长台钓加拿大/20240805 英文无字幕 阿里这小子在水城威尼斯发来问候/audio_vocals.wav'
    while True:
        text = input('请输入：')
        tts(text, f'playground/{text}.wav', target_language='中文')
        
