import shutil
import os
from loguru import logger
import time
from utils.utils import save_wav, normalize_wav
import torch
import gc

# Biến toàn cục
auto_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mdx_separator = None

def get_mdx_separator(model_name="UVR-MDX-NET-Inst_HQ_3.onnx", output_dir="."):
    """Khởi tạo hoặc trả về instance Audio-Separator (MDX-Net)"""
    global mdx_separator
    try:
        from audio_separator.separator import Separator
        if mdx_separator is None:
            logger.info(f"Khởi tạo Audio-Separator với model: {model_name}")
            mdx_separator = Separator(
                output_dir=output_dir,
                output_format="wav",
                normalization_threshold=0.9,
                sample_rate=48000 # Khớp với DeepFilterNet
            )
            mdx_separator.load_model(model_name)
        return mdx_separator
    except ImportError:
        logger.error("Thư viện 'audio-separator' chưa được cài đặt. Vui lòng cài đặt: pip install audio-separator[gpu]")
        return None
    except Exception as e:
        logger.error(f"Lỗi khởi tạo Audio-Separator: {e}")
        return None

def separate_audio(folder: str, model_name: str = "UVR-MDX-NET-Inst_HQ_3.onnx", device: str = 'auto', progress: bool = True,
                   shifts: int = 0) -> None:
    """
    Tách âm thanh bằng MDX-Net (Audio-Separator).
    """
    audio_path = os.path.join(folder, 'audio.wav')
    if not os.path.exists(audio_path):
        return None, None
        
    vocal_output_path = os.path.join(folder, 'audio_vocals.wav')
    instruments_output_path = os.path.join(folder, 'audio_instruments.wav')

    if os.path.exists(vocal_output_path) and os.path.exists(instruments_output_path):
        logger.info(f"Đã có file tách âm thanh: {folder}")
        return vocal_output_path, instruments_output_path

    logger.info(f"Sử dụng MDX-Net để tách âm thanh: {folder}")
    t_start = time.time()
    
    sep = get_mdx_separator(model_name, output_dir=folder)
    if sep is None:
        raise Exception("Không thể khởi tạo MDX Separator. Đảm bảo đã cài đặt audio-separator.")

    # Tiến hành tách
    output_files = sep.separate(audio_path)
    
    # MDX-Net Inst_HQ_3 thường tạo ra 2 file: Instrumental và Vocals
    for file_path in output_files:
        filename = os.path.basename(file_path)
        if "Vocals" in filename:
            shutil.move(os.path.join(folder, filename), vocal_output_path)
        elif "Instrumental" in filename:
            shutil.move(os.path.join(folder, filename), instruments_output_path)
    
    t_end = time.time()
    logger.info(f"Tách âm thanh hoàn tất trong {t_end - t_start:.2f}s")
    
    # Xóa file gốc để tiết kiệm dung lượng
    try:
        os.remove(audio_path)
        logger.info(f'Đã xóa file gốc: {audio_path}')
    except:
        pass
        
    return vocal_output_path, instruments_output_path

def release_model():
    """
    Giải phóng tài nguyên MDX Separator
    """
    global mdx_separator
    if mdx_separator is not None:
        logger.info('Đang giải phóng tài nguyên MDX Separator...')
        mdx_separator = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def extract_audio_from_video(folder: str, video_path: str = None) -> bool:
    """
    Trích xuất âm thanh từ video.
    """
    if video_path is None:
        video_path = os.path.join(folder, 'download.mp4')
        
    if not os.path.exists(video_path):
        logger.error(f"Không tìm thấy video để trích xuất âm thanh: {video_path}")
        return False
        
    audio_path = os.path.join(folder, 'audio.wav')
    if os.path.exists(audio_path):
        return True

    logger.info(f'Đang trích xuất âm thanh từ video: {video_path} -> {audio_path}')
    os.system(f'ffmpeg -loglevel error -i "{video_path}" -vn -acodec pcm_s16le -ar 48000 -ac 2 "{audio_path}"')
    
    time.sleep(0.5)
    return os.path.exists(audio_path)

def separate_all_audio_under_folder(root_folder: str, model_name: str = "UVR-MDX-NET-Inst_HQ_3.onnx", device: str = 'auto',
                                    progress: bool = True, shifts: int = 0, video_path: str = None) -> None:
    """
    Xử lý tách tất cả âm thanh trong thư mục.
    """
    vocal_output_path, instruments_output_path = None, None

    try:
        # Trường hợp 1: Có video_path cụ thể
        if video_path and os.path.exists(video_path):
            if not os.path.exists(os.path.join(root_folder, 'audio_vocals.wav')):
                extract_audio_from_video(root_folder, video_path)
                vocal_output_path, instruments_output_path = separate_audio(root_folder, model_name)
            else:
                vocal_output_path = os.path.join(root_folder, 'audio_vocals.wav')
                instruments_output_path = os.path.join(root_folder, 'audio_instruments.wav')
            return f'Xử lý hoàn tất: {root_folder}', vocal_output_path, instruments_output_path

        # Trường hợp 2: Duyệt folder
        for subdir, dirs, files in os.walk(root_folder):
            if 'download.mp4' not in files and not video_path:
                continue
            
            current_video = video_path if video_path else os.path.join(subdir, 'download.mp4')
            
            if 'audio.wav' not in files and 'audio_vocals.wav' not in files:
                extract_audio_from_video(subdir, current_video)
            
            if 'audio_vocals.wav' not in files:
                vocal_output_path, instruments_output_path = separate_audio(subdir, model_name)
            else:
                vocal_output_path = os.path.join(subdir, 'audio_vocals.wav')
                instruments_output_path = os.path.join(subdir, 'audio_instruments.wav')

        return f'Tất cả đã xử lý xong: {root_folder}', vocal_output_path, instruments_output_path

    except Exception as e:
        logger.error(f'Lỗi trong quá trình tách âm thanh: {str(e)}')
        release_model()
        raise

if __name__ == '__main__':
    separate_all_audio_under_folder("outputs")