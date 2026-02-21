import os
import uuid
import shutil
import json
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
from loguru import logger

# Add src to path to import modules
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from modules.asr.manager import transcribe_audio
    logger.info("Successfully loaded server-side ASR module (WhisperX/FunASR)")
except ImportError as e:
    logger.warning(f"ASR module loading failed: {e}. Server-side ASR will be unavailable.")
    transcribe_audio = None

# Giả định các module đã có sẵn trong project
# Chúng ta sẽ cần tích hợp với core logic của Linly-Dubbing
# từ src/core/engine.py hoặc các module riêng lẻ

app = FastAPI(title="Linly-Dubbing Heavy-Client API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "outputs/api_chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

class DubResponse(BaseModel):
    task_id: str
    dubbed_audio_url: str

def cleanup_task(task_folder: str):
    """Dọn dẹp sau khi xử lý xong"""
    # Tùy chọn: Xóa sau 1 giờ hoặc giữ lại tùy nhu cầu
    pass

@app.post("/api/dub-chunk", response_model=DubResponse)
async def dub_chunk(
    background_tasks: BackgroundTasks,
    audio_vocals: UploadFile = File(...),
    audio_instruments: UploadFile = File(...),
    target_lang: str = Form("vi"),
    voice: str = Form("female")
):
    task_id = str(uuid.uuid4())
    task_folder = os.path.join(OUTPUT_DIR, task_id)
    os.makedirs(task_folder, exist_ok=True)
    
    # Lưu các file nhận được
    vocals_path = os.path.join(task_folder, "audio_vocals.wav")
    instruments_path = os.path.join(task_folder, "audio_instruments.wav")
    
    with open(vocals_path, "wb") as f: shutil.copyfileobj(audio_vocals.file, f)
    with open(instruments_path, "wb") as f: shutil.copyfileobj(audio_instruments.file, f)

    try:
        # 1. Server-side ASR (WhisperX) - Always run since client-side is removed
        if transcribe_audio:
            logger.info(f"Task {task_id}: Running server-side ASR (WhisperX/FunASR)...")
            # Using 'small' model and disabling diarization to prevent OOM and speed up processing
            transcript_data = transcribe_audio('WhisperX', task_folder, model_name='small', device='auto', diarization=False)
        else:
            logger.error(f"Task {task_id}: Server-side ASR module not available")
            transcript_data = {"text": "[ASR MISSING]"}
            
        # 2. Dịch thuật (Translation)
        # TODO: Gọi module translation của Linly-Dubbing
        # Ví dụ: translated_data = translate_transcript(transcript_data, target_lang)
        
        # 3. Tạo giọng nói (TTS)
        # TODO: Gọi module TTS của Linly-Dubbing dựa trên mốc thời gian của vocals
        # output_vocals_dubbed = generate_tts(translated_data, voice)
        
        # 4. Trộn với nhạc nền (Remix)
        # TODO: Coi như đã có file dubbed_vocals.wav
        # combined_audio = mix_audio(dubbed_vocals, instruments_path)
        
        dubbed_filename = "dubbed_result.wav"
        dubbed_path = os.path.join(task_folder, dubbed_filename)
        
        # Ở đây mình giả lập việc copy (trong thực tế sẽ là kết quả của TTS + Mix)
        shutil.copy(vocals_path, dubbed_path) # Placeholder
        
        dubbed_audio_url = f"/static/{task_id}/{dubbed_filename}"
        
        return DubResponse(
            task_id=task_id,
            dubbed_audio_url=dubbed_audio_url
        )

    except Exception as e:
        logger.error(f"Dubbing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
