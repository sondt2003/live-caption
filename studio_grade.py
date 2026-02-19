import sys
import os
import shutil

# Thêm đường dẫn src vào hệ thống
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from core.engine import engine_run

# Test video path
video_path = "/home/dangson/workspace/live-caption/video/video3.mp4"
output_dir = "outputs/studio_grade"

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

print("Starting Studio-Grade Pipeline Test...")
status, output_video = engine_run(
    root_folder=output_dir,          # Thư mục gốc lưu trữ kết quả (outputs/studio_grade)
    url=None,                         # URL video (YouTube/Bilibili), set None nếu dùng file video cục bộ
    video_file=video_path,            # Đường dẫn tới file video cục bộ
    
    # Phương pháp dịch thuật: 'LLM', 'Google Translate', 'OpenAI', 'Ollama', 'Qwen', 'Ernie', 'Bing'
    translation_method='Google Translate', 
    translation_target_language='vi', # Ngôn ngữ đích cho dịch thuật (ISO code: vi, en, zh-cn, ja, ko...)
    
    # Phương pháp TTS: 'EdgeTTS', 'vits', 'xtts', 'cosyvoice'
    tts_target_language='vi',         # Ngôn ngữ đích cho TTS
    tts_method='EdgeTTS',             
    
    # Phương pháp ASR (Nhận diện giọng nói): 'WhisperX', 'FunASR'
    asr_method='WhisperX',            
    # Model WhisperX: 'tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3'
    whisper_model='small',            
    batch_size=8,                     # Kích thước batch cho ASR (tăng nếu có nhiều VRAM)
    diarization=True,                 # True: Phân biệt người nói, False: Không phân biệt
    
    # Model tách nhạc/vocal: 'htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'htdemucs_mmi'
    demucs_model='htdemucs_ft',       
    
    # Độ phân giải đầu ra: '720p', '1080p', '4k', '2k'
    target_resolution='1080p'         
)

print(f"Test Status: {status}")
print(f"Final Video: {output_video}")
