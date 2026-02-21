import os
import requests
import json
from loguru import logger
from ..base import BaseTTS

class MinimaxProvider(BaseTTS):
    def __init__(self, api_key=None, api_url=None):
        self.api_key = api_key or os.environ.get('XI_API_KEY')
        # Minimax endpoints on AI33 usually start with v1m
        # Default base URL: https://api.ai33.pro
        self.api_url = api_url or os.environ.get('ELEVENLABS_API_URL', "https://api.ai33.pro")
        
        # Ensure URL doesn't end with slash
        if self.api_url.endswith("/"):
            self.api_url = self.api_url[:-1]
            
        # Clean up v1 suffix if present, as we need root for v1m
        if self.api_url.endswith("/v1"):
            self.api_url = self.api_url[:-3]
            
        self.voice_cache = {} # Cache created voice_ids: { "speaker_name": "voice_id" }
        
        if not self.api_key:
            logger.warning("XI_API_KEY not found in environment variables. Minimax TTS may fail.")

    def generate(self, text: str, output_path: str, **kwargs) -> None:
        if os.path.exists(output_path):
            return

        voice_id = kwargs.get('voice')
        speaker_wav = kwargs.get('speaker_wav')
        
        # 1. Automatic Voice Cloning
        if not voice_id and speaker_wav and os.path.exists(speaker_wav):
            voice_id = self._get_or_create_cloned_voice(speaker_wav)
            
        # 2. Fallback to default if still None (using Minimax default voice ID from docs or sample)
        if not voice_id:
            voice_id = 'speech-01-hd' # Generic fallback model/voice? 
            # Actually Minimax uses specific voice_ids like "209533299589184"
            # Let's use a known one or just hope model default works.
            # Using the one from the example: "Graceful Lady" = "226893671006276"
            voice_id = "226893671006276" 

        model_id = kwargs.get('model_id', 'speech-01-hd') # or speech-2.6-hd
        
        tts_url = f"{self.api_url}/v1m/task/text-to-speech"
        
        headers = {
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model": model_id,
            "voice_setting": {
                "voice_id": voice_id,
                "vol": 1,
                "pitch": 0,
                "speed": 1
            },
            "language_boost": "Auto",
            "with_transcript": False
        }
        
        try:
            logger.info(f"Minimax Request: {tts_url} | Voice: {voice_id}")
            response = requests.post(tts_url, json=data, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"Minimax TTS Request Failed: {response.status_code} - {response.text}")
                raise Exception(f"API Error: {response.text}")

            resp_json = response.json()
            
            if not resp_json.get("success"):
                 raise Exception(f"Minimax API returned success=false: {resp_json}")

            # Minimax is typically Async? Docs say "Request we POST to your webhook... or you can polling"
            if "task_id" in resp_json:
                task_id = resp_json["task_id"]
                logger.info(f"Minimax Async Task initiated: {task_id}. Polling for result...")
                self._poll_and_download(task_id, output_path)
            else:
                logger.error(f"Unknown JSON response structure (no task_id): {resp_json}")
                raise Exception(f"Unknown JSON response: {resp_json}")
            
        except Exception as e:
            logger.error(f"Minimax TTS Exception: {str(e)}")
            raise

    def _get_or_create_cloned_voice(self, speaker_wav):
        """
        Automatically creates a voice from the speaker_wav using Minimax clone endpoint.
        """
        speaker_name = os.path.splitext(os.path.basename(speaker_wav))[0]
        
        if speaker_name in self.voice_cache:
            return self.voice_cache[speaker_name]
            
        logger.info(f"Auto-Cloning voice for speaker: {speaker_name} from {speaker_wav}")
        
        clone_url = f"{self.api_url}/v1m/voice/clone"
        headers = {"xi-api-key": self.api_key}
        
        # Minimax Clone expects multipart/form-data
        # file: @audio.mp3 (Max 20MB)
        # voice_name: string
        # preview_text: string (optional?)
        # language_tag: string
        # need_noise_reduction: boolean
        
        # IMPORTANT: Minimax documentation says "file: File only accept: audio.mp3"
        # We might need to convert valid WAV (from our pipeline) BACK to MP3 for upload!
        # Or maybe it checks extension. Let's try sending as mp3 or converting.
        
        temp_mp3 = speaker_wav.replace(".wav", ".mp3")
        conversion_needed = not speaker_wav.lower().endswith('.mp3')
        
        upload_file_path = speaker_wav
        
        if conversion_needed:
             try:
                from pydub import AudioSegment
                logger.info("Converting WAV to MP3 for Minimax cloning...")
                audio = AudioSegment.from_wav(speaker_wav)
                audio.export(temp_mp3, format="mp3")
                upload_file_path = temp_mp3
             except Exception as e:
                logger.warning(f"Could not convert to MP3 for upload: {e}. Trying raw WAV.")
                # We'll try uploading the WAV anyway, maybe docs are strict but API is loose.
        
        files_data = [
            ('file', (os.path.basename(upload_file_path), open(upload_file_path, 'rb'), 'audio/mpeg'))
        ]
        
        data = {
            "voice_name": f"Clone-{speaker_name}-{self._generate_short_hash()}",
            "language_tag": "English" # Or "Auto"? Docs example says "English". Let's use English or just leave it.
            # "preview_text": "Hello world"
        }
        
        try:
            response = requests.post(clone_url, headers=headers, data=data, files=files_data)
            if response.status_code == 200:
                res_json = response.json()
                if res_json.get("success"):
                    voice_id = res_json.get('cloned_voice_id') or res_json.get('voice_id')
                    # Documentation says 'cloned_voice_id' in one place, 'voice_id' in list.
                    # Example response: { "success": true, "cloned_voice_id": int_voice_id }
                    
                    if not voice_id and 'data' in res_json:
                        # Sometimes data is nested
                        voice_id = res_json['data'].get('voice_id')

                    if voice_id:
                        logger.info(f"Successfully cloned voice '{data['voice_name']}' with ID: {voice_id}")
                        self.voice_cache[speaker_name] = str(voice_id) # Ensure string
                        return str(voice_id)
                
                logger.error(f"Clone returned 200 but failed/no ID: {res_json}")
                return None
            else:
                logger.warning(f"Failed to clone voice (Minimax): {response.status_code} - {response.text}")
                return None
        except Exception as e:
             logger.warning(f"Exception during Minimax voice cloning: {e}")
             return None
        finally:
            for _, (name, f, mime) in files_data:
                f.close()
            # Clean up temp mp3 if created
            if conversion_needed and os.path.exists(temp_mp3):
                os.remove(temp_mp3)

    def cleanup_voices(self):
        """
        Deletes all auto-created voices from Minimax account.
        """
        if not self.voice_cache:
            return

        logger.info(f"Cleaning up {len(self.voice_cache)} auto-created Minimax voices...")
        headers = {"xi-api-key": self.api_key}

        for name, voice_id in list(self.voice_cache.items()):
            delete_url = f"{self.api_url}/v1m/voice/clone/{voice_id}"
            try:
                logger.info(f"Deleting Minimax voice '{name}' ({voice_id})...")
                response = requests.delete(delete_url, headers=headers)
                if response.status_code == 200:
                    logger.info(f"Successfully deleted voice {voice_id}")
                else:
                    logger.warning(f"Failed to delete voice {voice_id}: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Error deleting voice {voice_id}: {e}")
        
        self.voice_cache.clear()

    def _generate_short_hash(self):
        import time
        return hex(int(time.time()))[2:][-6:]

    def _poll_and_download(self, task_id, output_path, timeout=120, interval=2):
        import time
        start_time = time.time()
        
        # Polling Endpoint: GET /v1/task/{task_id} OR /v1m/task/{task_id}?
        # Docs say: "Request we POST... or you can polling the Common / GET Task"
        # Example polling URL: https://api.ai33.pro/v1/task/$task_id
        # Note: It seems to use the common /v1/task endpoint, not v1m specific for polling?
        # Let's verify. The "Get task" section shows: curl "https://api.ai33.pro/v1/task/$task_id"
        
        task_url = f"{self.api_url}/v1/task/{task_id}"
        headers = {"xi-api-key": self.api_key}

        while time.time() - start_time < timeout:
            try:
                r = requests.get(task_url, headers=headers)
                if r.status_code != 200:
                    logger.warning(f"Polling warning: {r.status_code} - {r.text}")
                    time.sleep(interval)
                    continue
                
                data = r.json()
                status = data.get("status")
                
                if status == "done":
                    # Success
                    metadata = data.get("metadata", {})
                    # Minimax metadata: audio_url, srt_url
                    audio_url = metadata.get("audio_url")
                    
                    if not audio_url:
                        # Check "data" field if structure differs
                        if "data" in data and "audio_url" in data["data"]:
                             audio_url = data["data"]["audio_url"]
                    
                    if not audio_url:
                        raise Exception(f"Task done but no audio_url found. Data: {data}")
                    
                    logger.info(f"Task finished. Downloading audio from {audio_url}")
                    self._download_file(audio_url, output_path)
                    return
                
                elif status == "failed" or status == "error":
                    error_msg = data.get("error_message", "Unknown error")
                    raise Exception(f"Task failed: {error_msg}")
                
                else:
                    # pending, doing, or processing
                    time.sleep(interval)
            except Exception as e:
                logger.warning(f"Polling error: {e}")
                time.sleep(interval)
        
        raise TimeoutError(f"Polling timed out after {timeout} seconds for task {task_id}")

    def _download_file(self, url, output_path):
        # Same as ElevenLabs: download and convert to WAV if needed
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            temp_path = output_path + ".temp"
            with open(temp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Convert to WAV
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(temp_path)
                audio.export(output_path, format="wav")
                logger.info(f"Downloaded and converted audio to {output_path}")
            except Exception as e:
                logger.error(f"Failed to convert audio: {e}. Falling back to raw file.")
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_path, output_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            raise Exception(f"Failed to download audio: {r.status_code}")
