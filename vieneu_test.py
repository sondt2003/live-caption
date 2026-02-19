import argparse
import os
import sys
import torch
import numpy as np
import subprocess
from loguru import logger

# Thêm đường dẫn src vào hệ thống nếu cần
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

try:
    from vieneu import VieNeuTTS
    import soundfile as sf
    VIENEU_AVAILABLE = True
except ImportError:
    VIENEU_AVAILABLE = False
    logger.error("Thư viện 'vieneu' chưa được cài đặt.")

def extract_audio_from_video(video_path, output_wav):
    """Trích xuất âm thanh từ video sang file wav cho việc clone giọng."""
    logger.info(f"Đang trích xuất âm thanh từ video: {video_path}")
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '24000', '-ac', '1',
        output_wav, '-y'
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi trích xuất âm thanh: {e.stderr.decode()}")
        raise e

def main():
    if not VIENEU_AVAILABLE:
        return

    parser = argparse.ArgumentParser(description="VieNeu-TTS Voice Cloning CLI")
    parser.add_argument("--video", type=str, help="Đường dẫn tới file video (.mp4) hoặc audio (.wav) để clone giọng")
    parser.add_argument("--text", type=str, required=True, help="Nội dung văn bản muốn nói")
    parser.add_argument("--output", type=str, default="outputs/vieneu_cloned.wav", help="Đường dẫn lưu file kết quả")
    parser.add_argument("--model", type=str, default="pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf", help="Hugging Face model ID")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị chạy (cuda/cpu)")
    
    args = parser.parse_args()

    # --- Cấu hình ---
    input_path = args.video
    target_text = args.text
    output_path = args.output
    model_id = args.model
    device = args.device
    
    logger.info("-" * 50)
    logger.info(f"Yêu cầu Clone Voice (VieNeu) từ: {input_path if input_path else 'Mặc định'}")
    logger.info(f"Nội dung muốn nói: {target_text}")
    logger.info(f"Thiết bị: {device}")
    logger.info("-" * 50)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Xử lý file đầu vào
    reference_wav = input_path
    if input_path and input_path.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        video_base = os.path.splitext(os.path.basename(input_path))[0]
        reference_wav = os.path.join(temp_dir, f"ref_vieneu_{video_base}.wav")
        
        if not os.path.exists(reference_wav):
            extract_audio_from_video(input_path, reference_wav)

    try:
        # Khởi tạo mô hình VieNeuTTS
        logger.info(f"Đang tải mô hình: {model_id}...")
        # VieNeuTTS sử dụng backbone_repo và backbone_device
        model = VieNeuTTS(
            backbone_repo=model_id,
            backbone_device=device,
            codec_device=device
        )
        
        # Tạo giọng nói
        logger.info("Đang thực hiện tổng hợp giọng nói (infer)...")
        # infer trả về numpy array
        audio_np = model.infer(
            text=target_text,
            ref_audio=reference_wav if reference_wav and os.path.exists(reference_wav) else None
        )
        
        # Lưu file sử dụng soundfile (tần số lấy mẫu mặc định của VieNeu thường là 24000)
        sf.write(output_path, audio_np, 24000)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"THÀNH CÔNG! File đã được lưu tại: {output_path}")
            print(f"\nKết quả: {os.path.abspath(output_path)}")
        else:
            logger.error("Quá trình tổng hợp thất bại.")
            
    except Exception as e:
        logger.exception(f"Gặp lỗi khi chạy VieNeu: {e}")

if __name__ == "__main__":
    main()
