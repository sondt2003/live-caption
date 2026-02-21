import argparse
import os
from dotenv import load_dotenv
from src.core.engine import engine_run
from loguru import logger
import warnings
import torch

# Suppress TorchAudio/Torio warnings
warnings.filterwarnings("ignore", module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*orio.io.*deprecated.*")
warnings.filterwarnings("ignore", message=".*In 2.9, this function's implementation will be changed.*")
warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")
warnings.filterwarnings("ignore", message=".*torchaudio._backend.list_audio_backends.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.custom_fwd.*")
warnings.filterwarnings("ignore", message=".*CategoricalEncoder.expect_len was never called.*")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Studio Grade Video Dubbing')
    parser.add_argument('--video_file', type=str, required=True, help='Path to video file')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
   
    parser.add_argument('--separator_model', type=str, default='UVR-MDX-NET-Inst_HQ_3.onnx', help='Separator model')
  
    parser.add_argument('--whisper_model', type=str, default='small', help='Whisper model size')
    parser.add_argument('--batch_size', type=int, default=4, help='Whisper batch size')
  
    parser.add_argument('--diarization', action='store_true', help='Enable speaker diarization')
   
    parser.add_argument('--target_resolution', type=str, default='original', help='Target resolution')
    
    parser.add_argument('--video_volume', type=float, default=1.0, help='Original video volume')
 
    parser.add_argument('--audio_only', action='store_true', help='Only generate audio')
  
    parser.add_argument('--asr_method', type=str, default='google', choices=['google', 'whisperx'], help='ASR method')
    parser.add_argument('--google_api_key', type=str, default=None, help='Google Speech API Key (v2)')
    parser.add_argument('--groq_api_key', type=str, default=None, help='Groq API Key for translation')
  
    parser.add_argument('--language', type=str, default=None, help='Original language (None for Auto)')
    parser.add_argument('--translation_method', type=str, default='google', choices=['google', 'groq', 'baidu', 'ollama', 'llm', 'gemini'], help='Translation method to use')
    parser.add_argument('--target_language', type=str, default='vi', help='Target language code')

    args = parser.parse_args()

    # Set up environment variables from args if provided
    if args.google_api_key:
        os.environ['GOOGLE_API_KEY'] = args.google_api_key
    if args.groq_api_key:
        os.environ['GROQ_API_KEY'] = args.groq_api_key

    msg, output_video = engine_run(
        root_folder=args.output_dir,
        video_file=args.video_file,
        separator_model=args.separator_model,
        whisper_model=args.whisper_model,
        batch_size=args.batch_size,
        diarization=args.diarization,
        target_resolution=args.target_resolution,
        video_volume=args.video_volume,
        audio_only=args.audio_only,
        language=args.language,
        asr_method=args.asr_method,
        google_key=args.google_api_key or os.getenv('GOOGLE_API_KEY'),
        translation_method=args.translation_method,
        translation_target_language=args.target_language,
        tts_target_language=args.target_language # FIX: Pass target language to TTS
    )

    print(f"Test Status: {msg}")
    if "thành công" not in msg.lower():
        print(f"Error Message: {output_video}") 
    print(f"Final Video: {output_video}")

if __name__ == "__main__":
    main()
