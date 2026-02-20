
import os
import json
import subprocess
from loguru import logger
from ..base import BaseTTS

class XTTSProvider(BaseTTS):
    def __init__(self):
        self.venv_path = os.path.abspath(os.path.join(os.getcwd(), "envs/venv_xtts/bin/python"))
        self.bridge_path = os.path.abspath(os.path.join(os.getcwd(), "src/modules/tts/bridge.py"))
        self.supported_languages = [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 
            'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi'
        ]

    def generate(self, text: str, output_path: str, **kwargs) -> None:
        if os.path.exists(output_path):
            return

        target_language = kwargs.get('target_language', 'en').lower()
        if target_language == 'zh': target_language = 'zh-cn'
        
        if target_language not in self.supported_languages:
            logger.error(f"Language {target_language} not supported by XTTS. Supported: {self.supported_languages}")
            raise ValueError(f"Unsupported language: {target_language}")

        task = {
            "text": text,
            "output_path": output_path,
            "speaker_wav": kwargs.get("speaker_wav"),
            "language": target_language # Use the validated and converted language
        }
        self.generate_batch([task])

    def generate_batch(self, tasks: list) -> None:
        # Filter out existing files and prepare tasks for the bridge
        pending_tasks = []
        for t in tasks:
            if not os.path.exists(t["output_path"]):
                # Validate and convert language for each task
                task_language = t.get("language", "en").lower()
                if task_language == 'zh': task_language = 'zh-cn'

                if task_language not in self.supported_languages:
                    logger.error(f"Language {task_language} not supported by XTTS. Supported: {self.supported_languages}")
                    raise ValueError(f"Unsupported language: {task_language}")

                task_params = {
                    "text": t["text"],
                    "output_path": t["output_path"],
                    "speaker_wav": t.get("speaker_wav"),
                    "language": task_language
                }
                pending_tasks.append(task_params)
        
        if not pending_tasks:
            return

        # Split into chunks (e.g. 10 segments at a time) for stability
        chunk_size = 10
        task_chunks = [pending_tasks[i:i + chunk_size] for i in range(0, len(pending_tasks), chunk_size)]

        logger.info(f"XTTS generating {len(pending_tasks)} segments in {len(task_chunks)} batches...")
        
        for idx, chunk in enumerate(task_chunks):
            logger.info(f"Processing XTTS batch {idx+1}/{len(task_chunks)} ({len(chunk)} segments)...")
            
            cmd = [
                self.venv_path,
                self.bridge_path,
                "--provider", "xtts",
                "--params", json.dumps({"tasks": chunk})
            ]
            
            env = os.environ.copy()
            env["COQUI_TOS_AGREED"] = "1"

            try:
                result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True, encoding="utf-8")
                stdout = result.stdout
                
                lines = stdout.strip().splitlines()
                successes = [l for l in lines if l.startswith("SUCCESS:")]
                
                if len(successes) > 0:
                    logger.info(f"XTTS Batch {idx+1} successful: {len(successes)} segments")
                else:
                    logger.error(f"XTTS bridge output missing SUCCESS markers. STDOUT: {stdout}")
                    raise Exception("Bridge communication error: SUCCESS marker not found.")
            except subprocess.CalledProcessError as e:
                logger.error(f"XTTS batch {idx+1} failed with exit code {e.returncode}")
                if e.stderr: logger.error(f"STDERR: {e.stderr}")
                raise Exception(f"XTTS subprocess failed: {e.stderr or e.stdout}")
            except Exception as e:
                logger.error(f"XTTS unexpected error: {str(e)}")
                raise e
