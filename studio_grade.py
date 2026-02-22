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
    parser.add_argument("--whisper_model", type=str, default="small", help="WhisperX model name.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for ASR.")
    parser.add_argument("--diarization",default=False, action="store_true", help="Enable speaker diarization.")
    parser.add_argument("--target_resolution", type=str, default="original", help="Output video resolution.")
    parser.add_argument("--asr_method", type=str, default="google", choices=["whisperx", "google"], help="ASR method to use.")
    parser.add_argument("--google_api_key", type=str, default="AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw", help="Google Speech API v2 key.")
    args = parser.parse_args()

    # Test video path
    video_path = args.video_file
    output_dir = "outputs"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting Studio-Grade Pipeline (TTS: {args.tts_method}, ASR: {args.asr_method}, Model: {args.whisper_model})...")
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
        whisper_model=args.whisper_model,            
        batch_size=args.batch_size,                     # Kích thước batch cho ASR (tăng nếu có nhiều VRAM)
        diarization=args.diarization,                 # True: Phân biệt người nói, False: Không phân biệt
        
        # Model tách nhạc/vocal: MDX-Net ONNX
        separator_model=args.separator_model,       
        
        # Độ phân giải đầu ra: 'original' (giữ nguyên), '720p', '1080p'
        target_resolution=args.target_resolution,
        
        # Âm lượng: 1.0 là mặc định, giảm xuống để bớt dính tiếng gốc (vocal leakage)
        video_volume=args.video_volume,
        
        # Tùy chọn chỉ xuất âm thanh
        audio_only=args.audio_only,
        language=args.language,
        asr_method=args.asr_method,
        google_key=args.google_api_key
    )

    print(f"Test Status: {msg}")
    if "thành công" not in msg.lower():
        print(f"Error Message: {output_video}") 
    print(f"Final Video: {output_video}")

if __name__ == "__main__":
    main()
