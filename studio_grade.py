import sys
import os
import shutil

# Thêm đường dẫn src vào hệ thống
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from core.engine import engine_run

import argparse

def main():
    parser = argparse.ArgumentParser(description="Studio-Grade Pipeline Test")
    parser.add_argument("--tts_method", type=str, default="edge", help="TTS method (auto, edge, minimax). 'auto' uses EdgeTTS by default.")
    parser.add_argument("--voice", type=str, default=None, help="Voice name or speaker_wav path for XTTS cloning.")
    parser.add_argument("--video_file", type=str, required=True, help="Path to input video file")
    parser.add_argument("--video_volume", type=float, default=0.5, help="Video vocal volume (0.0 - 1.0)")
    parser.add_argument("--separator_model", type=str, default="UVR-MDX-NET-Inst_HQ_3.onnx", help="Separation model (MDX-Net ONNX, e.g. UVR-MDX-NET-Inst_HQ_3.onnx)")
    parser.add_argument("--audio_only", action="store_true", help="Only generate audio, skip video synthesis")
    parser.add_argument("--language", type=str, default=None, help="ASR language code (e.g., 'en', 'vi'). Skips auto-detection if provided.")
    args = parser.parse_args()

    # Test video path
    video_path = args.video_file
    output_dir = "outputs"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting Studio-Grade Pipeline Test (TTS: {args.tts_method}, ASR: WhisperX)...")
    msg, output_video = engine_run(
        root_folder=output_dir,          # Thư mục gốc lưu trữ kết quả (outputs/studio_grade)
        url=None,                         # URL video (YouTube/Bilibili), set None nếu dùng file video cục bộ
        video_file=video_path,            # Đường dẫn tới file video cục bộ
        
        # Phương pháp dịch thuật: 'LLM', 'Google Translate', 'OpenAI', 'Ollama', 'Qwen', 'Ernie', 'Bing'
        translation_method='Google Translate', 
        translation_target_language='vi', # Ngôn ngữ đích cho dịch thuật (ISO code: vi, en, zh-cn, ja, ko...)
        
        # Phương pháp TTS: 'edge', 'minimax'
        tts_target_language='vi',         # Ngôn ngữ đích cho TTS
        tts_method=args.tts_method,
        voice=args.voice,
        
        # Model WhisperX: 'tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3'
        whisper_model='small',            
        batch_size=8,                     # Kích thước batch cho ASR (tăng nếu có nhiều VRAM)
        diarization=False,                 # True: Phân biệt người nói, False: Không phân biệt
        
        # Model tách nhạc/vocal: MDX-Net ONNX
        separator_model=args.separator_model,       
        
        # Độ phân giải đầu ra: 'original' (giữ nguyên), '720p', '1080p'
        target_resolution='original',
        
        # Âm lượng: 1.0 là mặc định, giảm xuống để bớt dính tiếng gốc (vocal leakage)
        video_volume=args.video_volume,
        
        # Tùy chọn chỉ xuất âm thanh
        audio_only=args.audio_only,
        language=args.language
    )

    print(f"Test Status: {msg}")
    if "thành công" not in msg.lower():
        print(f"Error Message: {output_video}") 
    print(f"Final Video: {output_video}")

if __name__ == "__main__":
    main()
