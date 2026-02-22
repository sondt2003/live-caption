import os
import requests
import json
from loguru import logger
from ..base import BaseTTS

class ElevenLabsProvider(BaseTTS):
    def __init__(self, api_key=None, api_url=None):
        self.api_key = api_key or os.environ.get('XI_API_KEY')
        # Default to AI33, but allow override to official ElevenLabs or others
        self.api_url = api_url or os.environ.get('ELEVENLABS_API_URL', "https://api.ai33.pro/v1")
        
        self.voice_cache = {} # Cache created voice_ids: { "speaker_name": "voice_id" }
        
        if not self.api_key:
            logger.warning("XI_API_KEY not found in environment variables. TTS may fail.")

    def generate_batch(self, tasks: list) -> None:
        """
        Generate multiple audio files in parallel using all available threads.
        """
        import concurrent.futures
        
        # Unlimited workers (one per task) for maximum throughput
        max_workers = len(tasks) if tasks else 1
        logger.info(f"ElevenLabs/AI33: Generating {len(tasks)} segments in parallel (max_workers={max_workers})...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for task in tasks:
                text = task.get("text")
                output_path = task.get("output_path")
                kwargs = {k: v for k, v in task.items() if k not in ["text", "output_path"]}
                futures.append(executor.submit(self.generate, text, output_path, **kwargs))
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"ElevenLabs batch task failed: {e}")

    def generate(self, text: str, output_path: str, **kwargs) -> None:
        if os.path.exists(output_path):
            return

        voice_id = kwargs.get('voice')
        speaker_wav = kwargs.get('speaker_wav')
        
        # 1. Automatic Voice Cloning (Like VieNeu)
        # If no specific voice ID is provided, but we have a speaker sample, try to clone it.
        if not voice_id and speaker_wav and os.path.exists(speaker_wav):
            voice_id = self._get_or_create_cloned_voice(speaker_wav)
            
        # 2. Fallback to default if still None
        if not voice_id:
            voice_id = '21m00Tcm4TlvDq8ikWAM' # Default to Rachel

        model_id = kwargs.get('model_id', 'eleven_multilingual_v2')
        
        # ... (rest of the generate logic is similar, just mapped to new vars) ...
        # Determine the base API URL (e.g., https://api.ai33.pro/v1)
        
        tts_url = f"{self.api_url}/text-to-speech/{voice_id}"
        
        headers = {
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        params = {
            "output_format": "mp3_44100_192" 
        }

        try:
            logger.info(f"ElevenLabs/AI33 Request: {tts_url} | Voice: {voice_id}")
            response = requests.post(tts_url, json=data, headers=headers, params=params)
            
            if response.status_code != 200:
                logger.error(f"TTS Request Failed: {response.status_code} - {response.text}")
                # Optimization: If 404/400 (maybe voice not found), try clearing cache or fallback?
                raise Exception(f"API Error: {response.text}")

            content_type = response.headers.get('Content-Type', '')
            
            # Scenario A: Direct Audio (Standard ElevenLabs)
            if 'audio' in content_type:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                logger.info(f"Audio received directly. Saved to {output_path}")
                return

            # Scenario B: Async Task (AI33 Wrapper)
            try:
                resp_json = response.json()
            except:
                logger.error(f"Unexpected response type: {content_type}")
                raise Exception(f"Unexpected response type: {content_type}")

            if "task_id" in resp_json:
                task_id = resp_json["task_id"]
                logger.info(f"Async Task initiated: {task_id}. Polling for result...")
                self._poll_and_download(task_id, output_path)
            else:
                logger.error(f"Unknown JSON response: {resp_json}")
                raise Exception(f"Unknown JSON response: {resp_json}")
                
        except Exception as e:
            logger.error(f"ElevenLabs TTS Exception: {str(e)}")
            raise

    def _get_or_create_cloned_voice(self, speaker_wav):
        """
        Automatically creates a voice from the speaker_wav if not already cached.
        """
        import hashlib
        
        # Check cache first
        # Use filename or hash of file path as key? 
        # Better: use the speaker name logic from path (e.g., SPEAKER/SPEAKER_01.wav)
        speaker_name = os.path.splitext(os.path.basename(speaker_wav))[0]
        
        if speaker_name in self.voice_cache:
            return self.voice_cache[speaker_name]
            
        logger.info(f"Auto-Cloning voice for speaker: {speaker_name} from {speaker_wav}")
        
        # Try v2 endpoint as v1 failed for listing and adding
        # AI33 likely uses v2 for voice management
        base_url = self.api_url.rsplit('/v1', 1)[0]
        if not base_url:
             base_url = self.api_url # fallback if no v1 found
             
        add_url = f"{base_url}/v2/voices/add"
        headers = {"xi-api-key": self.api_key}
        data = {
            "name": f"AutoClone-{speaker_name}-{self._generate_short_hash()}",
            "description": "Auto-generated by Studio-Grade Dubbing Pipeline"
        }
        
        files_data = [
            ('files', (os.path.basename(speaker_wav), open(speaker_wav, 'rb'), 'audio/mpeg'))
        ]
        
        try:
            response = requests.post(add_url, headers=headers, data=data, files=files_data)
            if response.status_code == 200:
                res_json = response.json()
                voice_id = res_json.get('voice_id')
                logger.info(f"Successfully created voice '{data['name']}' with ID: {voice_id}")
                self.voice_cache[speaker_name] = voice_id
                return voice_id
            elif response.status_code == 404:
                # Common issue with AI33 or reverse proxies without full API support
                logger.warning(f"Voice Cloning (Add Voice) not supported by this API endpoint: {add_url} (404). Falling back to default.")
                return None
            else:
                logger.warning(f"Failed to clone voice: {response.status_code} - {response.text}")
                return None
        except Exception as e:
             logger.warning(f"Exception during voice cloning: {e}")
             return None
        finally:
            for _, (name, f, mime) in files_data:
                f.close()

    def cleanup_voices(self):
        """
        Deletes all auto-created voices from the ElevenLabs/AI33 account to free up slots.
        """
        if not self.voice_cache:
            return

        logger.info(f"Cleaning up {len(self.voice_cache)} auto-created voices...")
        headers = {"xi-api-key": self.api_key}

        for name, voice_id in list(self.voice_cache.items()):
            delete_url = f"{self.api_url}/voices/{voice_id}"
            try:
                logger.info(f"Deleting voice '{name}' ({voice_id})...")
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
        
        task_url = f"{self.api_url.replace('/v1', '')}/v1/task/{task_id}" # Ensure /v1/task structure
        # Fallback if the user provided URL doesn't have /v1 or has it differently
        if '/v1' not in self.api_url:
             task_url = f"{self.api_url}/task/{task_id}"
        
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
                    audio_url = metadata.get("audio_url")
                    if not audio_url:
                        raise Exception("Task done but no audio_url found in metadata")
                    
                    logger.info(f"Task finished. Downloading audio from {audio_url}")
                    self._download_file(audio_url, output_path)
                    return
                
                elif status == "failed":
                    error_msg = data.get("error_message", "Unknown error")
                    raise Exception(f"Task failed: {error_msg}")
                
                else:
                    # pending or processing
                    time.sleep(interval)
            except Exception as e:
                logger.warning(f"Polling error: {e}")
                time.sleep(interval)
        
        raise TimeoutError(f"Polling timed out after {timeout} seconds for task {task_id}")

    def _download_file(self, url, output_path):
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            temp_path = output_path + ".temp"
            with open(temp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Convert to WAV using pydub if needed (ElevenLabs sends MP3 by default)
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(temp_path)
                audio.export(output_path, format="wav")
                logger.info(f"Downloaded and converted audio to {output_path}")
            except Exception as e:
                logger.error(f"Failed to convert audio: {e}. Falling back to raw file.")
                # Fallback: just rename, hope downstream can handle it if not strict WAV
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_path, output_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            raise Exception(f"Failed to download audio: {r.status_code}")
