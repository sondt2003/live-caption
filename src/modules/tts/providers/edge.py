import os
import sys
import subprocess
import torchaudio
from loguru import logger
from ..base import BaseTTS

class EdgeTTSProvider(BaseTTS):
    def __init__(self):
        self.language_map = {
            'vi': 'vi-VN-HoaiMyNeural',
            'zh-cn': 'zh-CN-XiaoxiaoNeural',
            'en': 'en-US-MichelleNeural',
            'ja': 'ja-JP-NanamiNeural',
            'yue': 'zh-HK-HiuMaanNeural',
            'ko': 'ko-KR-SunHiNeural',
        }
        self.edge_tts_path = os.path.join(os.path.dirname(sys.executable), 'edge-tts')

    def generate(self, text: str, output_path: str, **kwargs) -> None:
        if os.path.exists(output_path):
            return

        target_language = kwargs.get('target_language', 'vi').lower()
        voice = kwargs.get('voice')

        # Logic for voice selection
        if voice is None or voice in ['zh-CN-XiaoxiaoNeural', 'ja-JP-NanamiNeural']:
            voice = self.language_map.get(target_language, 'vi-VN-HoaiMyNeural')
        
        if ('vi' in target_language) and 'vi-VN' not in voice:
            voice = 'vi-VN-HoaiMyNeural'

        logger.info(f"Using EdgeTTS voice: {voice} for language: {target_language}")
        
        mp3_path = output_path.replace(".wav", ".mp3")
        for retry in range(3):
            try:
                result = subprocess.run(
                    [self.edge_tts_path, '--text', text, '--write-media', mp3_path, '--voice', voice],
                    capture_output=True, text=True
                )
                
                if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
                    audio, sr = torchaudio.load(mp3_path)
                    torchaudio.save(output_path, audio, sr)
                    os.remove(mp3_path)
                    break
                else:
                    logger.warning(f"EdgeTTS failed (retry {retry}): {result.stderr}")
            except Exception as e:
                logger.error(f"EdgeTTS unexpected error: {e}")
