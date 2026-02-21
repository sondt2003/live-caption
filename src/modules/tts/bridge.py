
import argparse
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="TTS Bridge for isolated environments")
    parser.add_argument("--provider", type=str, required=True, help="TTS provider (xtts, vieneu)")
    parser.add_argument("--params", type=str, required=True, help="JSON string of parameters")
    args = parser.parse_args()

    params = json.loads(args.params)
    
    # Batch processing support
    tasks = params.get("tasks")
    if not tasks:
        # Fallback to single task if "tasks" key is missing
        tasks = [params]

    try:
        if args.provider == "xtts":
            from TTS.api import TTS
            os.environ["COQUI_TOS_AGREED"] = "1"
            
            # Khởi tạo mô hình một lần cho toàn bộ batch
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
            
            for task in tasks:
                text = task.get("text")
                output_path = task.get("output_path")
                speaker_wav = task.get("speaker_wav")
                language = task.get("language", "en")
                
                tts.tts_to_file(
                    text=text,
                    speaker_wav=speaker_wav,
                    language=language,
                    file_path=output_path
                )
                print(f"SUCCESS:{output_path}")

        elif args.provider == "vieneu":
            from vieneu import Vieneu
            
            # Khởi tạo model MỘT LẦN duy nhất cho toàn bộ batch
            tts = Vieneu()
            try:
                for task in tasks:
                    text = task.get("text")
                    output_path = task.get("output_path")
                    ref_audio = task.get("ref_audio")
                    ref_text = task.get("ref_text")
                    voice_name = task.get("voice")
                    
                    if ref_audio and ref_text:
                        # Use cloned voice with reference
                        audio = tts.infer(text=text, ref_audio=ref_audio, ref_text=ref_text)
                    elif ref_audio:
                        # Use cloned voice without text (fallback)
                        audio = tts.infer(text=text, ref_audio=ref_audio)
                    elif voice_name:
                        # Use preset voice
                        voice_data = tts.get_preset_voice(voice_name)
                        audio = tts.infer(text=text, voice=voice_data)
                    else:
                        # Default voice
                        audio = tts.infer(text=text)
                        
                    tts.save(audio, output_path)
                    print(f"SUCCESS:{output_path}")
            finally:
                tts.close()

        else:
            print(f"Error: Unknown provider {args.provider}")
            sys.exit(1)

    except Exception as e:
        print(f"ERROR:{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
