import sys
import os
import argparse
import traceback
from tools.do_everything import do_everything
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description='Linly-Dubbing CLI')
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('--tts_method', default='EdgeTTS', choices=['xtts', 'bytedance', 'cosyvoice', 'EdgeTTS', 'vits'], help='TTS method to use')
    parser.add_argument('--translation_method', default='Google Translate', choices=['Google Translate', 'LLM', 'GPT', 'Ollama', 'Qwen'], help='Translation method')
    parser.add_argument('--target_language', default='Vietnamese', help='Target language for translation and TTS')
    parser.add_argument('--app_language', default='zh', choices=['zh', 'en'], help='App UI language (for logs)')
    parser.add_argument('--clean', action='store_true', help='Clean output folder before processing to ensure fresh run')
    
    args = parser.parse_args()
    
    video_path = args.video_path
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    root_folder = 'videos'
    os.makedirs(root_folder, exist_ok=True)
    
    # Calculate output folder path to clean it if requested
    import shutil
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(root_folder, base_name)
    
    if args.clean and os.path.exists(output_folder):
        print(f"Cleaning output folder: {output_folder}")
        shutil.rmtree(output_folder)
        
    print(f"--- Starting CLI Processing ---")
    print(f"Video File: {video_path}")
    print(f"TTS Method: {args.tts_method}")
    print(f"Target Language: {args.target_language}")
    print(f"-------------------------------")
    
    try:
        # We call do_everything with the parameters using args
        result, out_video = do_everything(
            root_folder=root_folder,
            url='', # URL is empty as we use local file
            video_file=video_path,
            num_videos=1,
            resolution='1080p',
            asr_method='WhisperX',
            whisper_model='small', # 'large' requires >4GB VRAM
            batch_size=4,          # Reduced for memory stability
            translation_method=args.translation_method,
            translation_target_language=args.target_language,
            tts_method=args.tts_method, 
            tts_target_language=args.target_language, 
            voice='vi-VN-HoaiMyNeural', # Default for EdgeTTS, others might ignore or use their own defaults
            subtitles=True,
            speed_up=1.00,
            fps=30
        )
        
        print(f"\nProcessing Result: {result}")
        if out_video:
            print(f"Final Output Video: {out_video}")
        else:
            print("No output video was generated.")
            
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
