import os
import torchaudio
from openai import OpenAI
from loguru import logger
from ..base import BaseTTS

class OpenAITTSProvider(BaseTTS):
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.base_url = base_url or os.environ.get('OPENAI_BASE_URL')
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.api_key else None

    def generate(self, text: str, output_path: str, **kwargs) -> None:
        if os.path.exists(output_path):
            return

        if not self.client:
            logger.error("OpenAI API Key not set.")
            raise ValueError("OpenAI API Key not set.")

        voice = kwargs.get('voice', 'alloy')
        model = kwargs.get('model', 'tts-1')

        logger.info(f"Using OpenAI TTS voice: {voice}, model: {model}")

        mp3_path = output_path.replace(".wav", ".mp3")
        try:
            response = self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text
            )
            response.stream_to_file(mp3_path)

            if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
                # Convert mp3 to wav
                audio, sr = torchaudio.load(mp3_path)
                torchaudio.save(output_path, audio, sr)
                os.remove(mp3_path)
                logger.info(f"OpenAI TTS successfully generated to {output_path}")
            else:
                logger.error("OpenAI TTS failed to save mp3 file.")
        except Exception as e:
            logger.error(f"OpenAI TTS unexpected error: {e}")
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
