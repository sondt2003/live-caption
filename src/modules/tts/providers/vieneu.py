
import os
import json
import subprocess
import sys
from loguru import logger
from ..base import BaseTTS

class VieNeuProvider(BaseTTS):
    def __init__(self):
        self.venv_path = os.path.abspath(os.path.join(os.getcwd(), "envs/venv_vieneu/bin/python"))
        self.bridge_path = os.path.abspath(os.path.join(os.getcwd(), "src/modules/tts/bridge.py"))

    def generate(self, text: str, output_path: str, **kwargs) -> None:
        if os.path.exists(output_path):
            return
        
        task = {
            "text": text,
            "output_path": output_path,
            "voice": kwargs.get("voice") or "Binh",
            "ref_audio": kwargs.get("speaker_wav") if kwargs.get("ref_text") else None,
            "ref_text": kwargs.get("ref_text")
        }
        self.generate_batch([task])

    def generate_batch(self, tasks: list) -> None:
        """
        Tasks is a list of dicts with keys: text, output_path, voice, speaker_wav, ref_text
        """
        # Filter out existing files
        pending_tasks = []
        for t in tasks:
            if not os.path.exists(t["output_path"]):
                task_params = {
                    "text": t["text"],
                    "output_path": t["output_path"],
                    "voice": t.get("voice") or "Binh",
                    "ref_audio": t.get("speaker_wav") if t.get("ref_text") else None,
                    "ref_text": t.get("ref_text")
                }
                pending_tasks.append(task_params)
        
        # Send all segments in one subprocess call
        logger.info(f"VieNeu-TTS generating {len(pending_tasks)} segments (isolated venv, load-once)...")
        
        cmd = [
            self.venv_path,
            self.bridge_path,
            "--provider", "vieneu",
             "--params", json.dumps({"tasks": pending_tasks})
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8")
            stdout = result.stdout
            
            lines = stdout.strip().splitlines()
            successes = [l for l in lines if l.startswith("SUCCESS:")]
            if len(successes) > 0:
                logger.info(f"VieNeu-TTS successful: {len(successes)} segments")
            else:
                logger.warning(f"VieNeu-TTS completed but no SUCCESS markers found.")

        except subprocess.CalledProcessError as e:
            logger.error(f"VieNeu-TTS failed with exit code {e.returncode}")
            raise Exception(f"VieNeu-TTS failed: {e.stderr or e.stdout}")
        except Exception as e:
            logger.error(f"VieNeu-TTS unexpected error: {str(e)}")
            raise e
