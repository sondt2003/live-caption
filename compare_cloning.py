import os
import sys
import shutil
sys.dont_write_bytecode = True
import subprocess
import argparse
from loguru import logger

# Thêm đường dẫn src vào hệ thống
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from utils.separation import separate_all_audio_under_folder
from modules.tts.xtts import tts as xtts_func
from modules.tts.cosyvoice import tts as cosy_func
from modules.tts.vits import tts as vits_func

logger.info("Khởi tạo script so sánh Voice Clone (Studio-Grade)...")

def get_audio_from_video(video_path, output_folder):
    """Trích xuất giọng nói sạch từ video để làm mẫu voice clone"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    vid_name = os.path.splitext(os.path.basename(video_path))[0]
    import re
    vid_name = re.sub(r'[\\/*?:"<>|]', '', vid_name)
    vid_folder = os.path.join(output_folder, vid_name)
    os.makedirs(vid_folder, exist_ok=True)
    
    dest_video = os.path.join(vid_folder, "download.mp4")
    if not os.path.exists(dest_video):
        shutil.copy(video_path, dest_video)
        
    logger.info(f"Đang tách giọng nói tham chiếu từ {video_path}...")
    try:
        # Sử dụng Demucs để tách giọng sạch (Studio-Grade)
        status, vocals_path, _ = separate_all_audio_under_folder(vid_folder, model_name='htdemucs_ft', device='auto', shifts=1)
        return vocals_path
    except Exception as e:
        logger.error(f"Lỗi khi tách tiếng (Demucs): {e}")
        audio_path = os.path.join(vid_folder, "audio.wav")
        subprocess.run(['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '24000', '-ac', '1', audio_path, '-y'], check=True)
        return audio_path

def merge_audio_video(video_path, audio_path, output_path):
    """Ghép âm thanh đã clone vào video gốc"""
    cmd = [
        'ffmpeg', '-i', video_path, '-i', audio_path,
        '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
        '-shortest', '-y', output_path
    ]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description='So sánh các mô hình Voice Cloning (Linly-Dubbing)')
    parser.add_argument('video_path', help='Đường dẫn video chứa giọng nói tham chiếu')
    parser.add_argument('--text', default="Xin chào, đây là bản thử nghiệm lồng tiếng AI chất lượng cao. Bạn thấy giọng của tôi có giống bản gốc không?", help='Văn bản để đọc thử')
    parser.add_argument('--output_dir', default='outputs/comparisons', help='Thư mục lưu kết quả')
    parser.add_argument('--lang', default='vi', help='Ngôn ngữ lồng tiếng (vi, en, ja...)')
    
    args = parser.parse_args()
    
    video_path = args.video_path
    text = args.text
    output_dir = args.output_dir
    lang = args.lang
    
    if not os.path.exists(video_path):
        print(f"Lỗi: Không tìm thấy file video tại {video_path}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Trích xuất giọng nói mẫu
    print(f"\n--- Bước 1: Trích xuất giọng nói tham chiếu (Vocal Reference) ---")
    speaker_wav = get_audio_from_video(video_path, os.path.join(output_dir, 'temp'))
    print(f"File tham chiếu sạch: {speaker_wav}")
    
    # 2. XTTS
    print(f"\n--- Bước 2: Thử nghiệm clone bằng XTTS (Gala-Standard) ---")
    xtts_wav = os.path.join(output_dir, 'result_xtts.wav')
    xtts_video = os.path.join(output_dir, 'result_xtts.mp4')
    try:
        xtts_func(text, xtts_wav, speaker_wav=speaker_wav, target_language=lang) 
        if os.path.exists(xtts_wav):
            merge_audio_video(video_path, xtts_wav, xtts_video)
            print(f"Thành công! Kết quả XTTS: {xtts_video}")
    except Exception as e:
        print(f"XTTS thất bại: {e}")

    # 3. CosyVoice
    print(f"\n--- Bước 3: Thử nghiệm clone bằng CosyVoice (High-Fidelity) ---")
    cosy_wav = os.path.join(output_dir, 'result_cosyvoice.wav')
    cosy_video = os.path.join(output_dir, 'result_cosyvoice.mp4')
    try:
        cosy_func(text, cosy_wav, speaker_wav=speaker_wav, target_language=lang)
        if os.path.exists(cosy_wav):
            merge_audio_video(video_path, cosy_wav, cosy_video)
            print(f"Thành công! Kết quả CosyVoice: {cosy_video}")
    except Exception as e:
        print(f"CosyVoice thất bại: {e}")

    # 4. VoxCPM (VITS - Chuyên dụng cho tiếng Việt)
    if lang == 'vi':
        print(f"\n--- Bước 4: Thử nghiệm clone bằng VoxCPM (Tối ưu cho tiếng Việt) ---")
        vox_wav = os.path.join(output_dir, 'result_voxcpm.wav')
        vox_video = os.path.join(output_dir, 'result_voxcpm.mp4')
        try:
            vits_func(text, vox_wav, speaker_wav=speaker_wav, target_language='vi') 
            if os.path.exists(vox_wav):
                merge_audio_video(video_path, vox_wav, vox_video)
                print(f"Thành công! Kết quả VoxCPM: {vox_video}")
        except Exception as e:
            print(f"VoxCPM thất bại: {e}")

    print(f"\n--- Hoàn tất so sánh! Tất cả kết quả được lưu tại: {output_dir} ---")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

