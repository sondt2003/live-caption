import sys
import os
import shutil

# Thêm đường dẫn src vào hệ thống
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from core.engine import engine_run

import argparse

def main():
    parser = argparse.ArgumentParser(description="Studio-Grade Pipeline Test")
    parser.add_argument("--tts_method", type=str, default="edge", help="TTS method (auto, edge, xtts, azure, openai). 'auto' uses XTTS for supported languages, EdgeTTS otherwise.")
    parser.add_argument("--voice", type=str, default=None, help="Voice name or speaker_wav path for XTTS cloning.")
    parser.add_argument("--asr_method", type=str, default="WhisperX", choices=['WhisperX', 'FunASR'], help="ASR method")
    parser.add_argument("--hardcode_subtitles", action="store_true", help="Hardcode subtitles into video (slower)")
    parser.add_argument("--video_volume", type=float, default=0.2, help="Background music volume")
    parser.add_argument("--shifts", type=int, default=1, help="Demucs shifts (default 1 for speed, increase for quality)")
    args = parser.parse_args()

    # Test video path
    video_path = "/home/dangson/workspace/live-caption/video/video-5.mp4"
    output_dir = "outputs"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting Studio-Grade Pipeline Test (TTS: {args.tts_method}, ASR: {args.asr_method})...")
    msg, output_video = engine_run(
        root_folder=output_dir,          # Thư mục gốc lưu trữ kết quả (outputs/studio_grade)
        url=None,                         # URL video (YouTube/Bilibili), set None nếu dùng file video cục bộ
        video_file=video_path,            # Đường dẫn tới file video cục bộ
        
        # Phương pháp dịch thuật: 'LLM', 'Google Translate', 'OpenAI', 'Ollama', 'Qwen', 'Ernie', 'Bing'
        translation_method='Google Translate', 
        translation_target_language='vi', # Ngôn ngữ đích cho dịch thuật (ISO code: vi, en, zh-cn, ja, ko...)
        
        # Phương pháp TTS: 'edge', 'gtts', 'azure', 'openai'
        tts_target_language='vi',         # Ngôn ngữ đích cho TTS
        tts_method=args.tts_method,
        voice=args.voice,
        subtitles=args.hardcode_subtitles,
        shifts=args.shifts,
        
        # Phương pháp ASR (Nhận diện giọng nói): 'WhisperX', 'FunASR'
        asr_method=args.asr_method,            
        # Model WhisperX: 'tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3'
        whisper_model='small',            
        batch_size=8,                     # Kích thước batch cho ASR (tăng nếu có nhiều VRAM)
        diarization=True,                 # True: Phân biệt người nói, False: Không phân biệt
        
        # Model tách nhạc/vocal: 'htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'htdemucs_mmi'
        demucs_model='htdemucs_ft',       
        
        # Độ phân giải đầu ra: '720p', '1080p', '4k', '2k'
        target_resolution='1080p',
        
        # Âm lượng: 1.0 là mặc định, giảm xuống để bớt dính tiếng gốc (vocal leakage)
        video_volume=args.video_volume
    )

    print(f"Test Status: {msg}")
    if "thành công" not in msg.lower():
        print(f"Error Message: {output_video}") 
    print(f"Final Video: {output_video}")

if __name__ == "__main__":
    main()
