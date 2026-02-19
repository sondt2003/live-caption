import os
import azure.cognitiveservices.speech as speechsdk
from loguru import logger
from ..base import BaseTTS

class AzureTTSProvider(BaseTTS):
    def __init__(self, api_key=None, region=None):
        self.api_key = api_key or os.environ.get('AZURE_SPEECH_KEY')
        self.region = region or os.environ.get('AZURE_SPEECH_REGION')

    def generate(self, text: str, output_path: str, **kwargs) -> None:
        if os.path.exists(output_path):
            return

        if not self.api_key or not self.region:
            logger.error("Azure Speech Key or Region not set.")
            raise ValueError("Azure Speech Key or Region not set.")

        target_language = kwargs.get('target_language', 'en-US')
        voice = kwargs.get('voice')

        # Simple voice selection if not provided
        if not voice:
            if 'vi' in target_language.lower():
                voice = 'vi-VN-HoaiMyNeural'
            elif 'zh' in target_language.lower():
                voice = 'zh-CN-XiaoxiaoNeural'
            else:
                voice = 'en-US-JennyNeural'

        logger.info(f"Using Azure TTS voice: {voice} for language: {target_language}")

        speech_config = speechsdk.SpeechConfig(subscription=self.api_key, region=self.region)
        speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)
        
        # Output directly to the final path
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        
        # Handle SSML if needed, but for now simple text is fine as a fallback or if voice is provided
        # If we want to support rate/pitch/volume from kwargs, we'd use SSML.
        # For a clean start, let's use the provided voice directly.
        
        result = speech_synthesizer.speak_text_async(text).get() if not voice else speech_synthesizer.start_speaking_text_async(text).get()
        
        # If voice is provided, it's better to use SSML or set the voice in speech_config
        if voice:
            speech_config.speech_synthesis_voice_name = voice
            speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
            result = speech_synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.info(f"Azure TTS successfully generated to {output_path}")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            logger.error(f"Azure TTS Canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                logger.error(f"Azure TTS Error details: {cancellation_details.error_details}")
            if os.path.exists(output_path):
                os.remove(output_path)
