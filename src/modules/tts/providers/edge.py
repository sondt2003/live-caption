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

    def generate_batch(self, tasks: list) -> None:
        """
        Generate multiple audio files in parallel.
        """
        import concurrent.futures
        
        # Use as many workers as there are tasks for maximum speed
        MAX_WORKERS = len(tasks) if tasks else 1
        
        logger.info(f"EdgeTTS: Generating {len(tasks)} segments in parallel (max_workers={MAX_WORKERS})...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for task in tasks:
                text = task.get("text")
                output_path = task.get("output_path")
                # Extract other kwargs
                kwargs = {k: v for k, v in task.items() if k not in ["text", "output_path"]}
                
                futures.append(executor.submit(self.generate, text, output_path, **kwargs))
            
            # Wait for all tasks to complete and handle exceptions
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"EdgeTTS batch task failed: {e}")

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
        import time 
        
        for retry in range(10):
            try:
                # Use --rate if provided (not implemented yet but good for future)
                # Use python -m edge_tts to be safe across environments
                cmd = [sys.executable, '-m', 'edge_tts', '--text', text, '--write-media', mp3_path, '--voice', voice]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True, text=True
                )
                
                if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
                    audio, sr = torchaudio.load(mp3_path)
                    torchaudio.save(output_path, audio, sr)
                    os.remove(mp3_path)
                    break
                else:
                    logger.warning(f"EdgeTTS failed (retry {retry}): {result.stderr}")
                    time.sleep(2) # Backoff before retry
            except Exception as e:
                logger.error(f"EdgeTTS unexpected error: {e}")
                time.sleep(2) # Backoff
