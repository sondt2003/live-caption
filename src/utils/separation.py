import shutil
from demucs.api import Separator
import os
from loguru import logger
import time
from utils.utils import save_wav, normalize_wav
import torch
import gc

# Biến toàn cục
auto_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
separator = None
model_loaded = False  # Cờ theo dõi trạng thái tải model
current_model_config = {}  # Biến lưu cấu hình model hiện tại


def init_demucs():
    """
    Khởi tạo mô hình Demucs.
    Nếu mô hình đã được khởi tạo, trả về ngay mà không tải lại.
    """
    global separator, model_loaded
    if not model_loaded:
        separator = load_model()
        model_loaded = True
    else:
        logger.info("Mô hình Demucs đã được tải, bỏ qua khởi tạo")


def load_model(model_name: str = "htdemucs_ft", device: str = 'auto', progress: bool = True,
               shifts: int = 1) -> Separator:
    """
    Tải mô hình Demucs.
    Nếu mô hình cùng cấu hình đã được tải, sử dụng lại mô hình hiện có.
    """
    global separator, model_loaded, current_model_config

    if separator is not None:
        # 检查是否需要重新加载模型（配置不同）
        requested_config = {
            'model_name': model_name,
            'device': 'auto' if device == 'auto' else device,
            'shifts': shifts
        }

        if current_model_config == requested_config:
            logger.info(f'Mô hình Demucs đã tải và cùng cấu hình, tái sử dụng')
            return separator
        else:
            logger.info(f'Cấu hình Demucs thay đổi, cần tải lại')
            # Giải phóng tài nguyên mô hình hiện tại
            release_model()

    logger.info(f'Đang tải mô hình Demucs: {model_name}')
    t_start = time.time()

    device_to_use = auto_device if device == 'auto' else device
    separator = Separator(model_name, device=device_to_use, progress=progress, shifts=shifts)

    # Lưu cấu hình model hiện tại
    current_model_config = {
        'model_name': model_name,
        'device': 'auto' if device == 'auto' else device,
        'shifts': shifts
    }

    model_loaded = True
    t_end = time.time()
    logger.info(f'Tải mô hình Demucs hoàn tất trong {t_end - t_start:.2f} giây')

    return separator


def release_model():
    """
    Giải phóng tài nguyên mô hình, tránh rò rỉ bộ nhớ
    """
    global separator, model_loaded, current_model_config

    if separator is not None:
        logger.info('Đang giải phóng tài nguyên mô hình Demucs...')
        # Xóa tham chiếu
        separator = None
        # Bắt buộc thu gom rác
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model_loaded = False
        current_model_config = {}
        logger.info('Đã giải phóng tài nguyên mô hình Demucs')


def separate_audio(folder: str, model_name: str = "htdemucs_ft", device: str = 'auto', progress: bool = True,
                   shifts: int = 1) -> None:
    """
    Tách file âm thanh
    """
    global separator
    audio_path = os.path.join(folder, 'audio.wav')
    if not os.path.exists(audio_path):
        return None, None
    vocal_output_path = os.path.join(folder, 'audio_vocals.wav')
    instruments_output_path = os.path.join(folder, 'audio_instruments.wav')
    # Đường dẫn cho 4-stem
    drums_output_path = os.path.join(folder, 'audio_drums.wav')
    bass_output_path = os.path.join(folder, 'audio_bass.wav')
    other_output_path = os.path.join(folder, 'audio_other.wav')

    if os.path.exists(vocal_output_path) and os.path.exists(instruments_output_path):
        logger.info(f'Đã tách âm thanh: {folder}')
        return vocal_output_path, instruments_output_path

    logger.info(f'Đang tách âm thanh: {folder}')

    try:
        # Đảm bảo mô hình đã tải và cấu hình đúng
        if not model_loaded or current_model_config.get('model_name') != model_name or \
                (current_model_config.get('device') == 'auto') != (device == 'auto') or \
                current_model_config.get('shifts') != shifts:
            load_model(model_name, device, progress, shifts)

        t_start = time.time()

        try:
            origin, separated = separator.separate_audio_file(audio_path)
        except Exception as e:
            logger.error(f'Lỗi tách âm thanh: {e}')
            # Thử tải lại mô hình nếu có lỗi
            release_model()
            load_model(model_name, device, progress, shifts)
            logger.info(f'Đã tải lại mô hình, đang thử lại...')
            origin, separated = separator.separate_audio_file(audio_path)

        t_end = time.time()
        logger.info(f'Tách âm thanh hoàn tất trong {t_end - t_start:.2f} giây')

        import torchaudio.functional as F
        target_sr = 48000
        demucs_sr = separator.samplerate

        # Resample về 48kHz để chuẩn hóa xử lý studio (khớp với DeepFilterNet)
        vocals_tensor = separated['vocals']
        if demucs_sr != target_sr:
            vocals_tensor = F.resample(vocals_tensor, demucs_sr, target_sr)
        vocals = vocals_tensor.numpy().T

        instr_tensor = (separated['drums'] + separated['bass'] + separated['other'])
        if demucs_sr != target_sr:
            instr_tensor = F.resample(instr_tensor, demucs_sr, target_sr)
        instruments = instr_tensor.numpy().T

        save_wav(vocals, vocal_output_path, sample_rate=target_sr)
        logger.info(f'已保存人声: {vocal_output_path} ({target_sr}Hz)')

        save_wav(instruments, instruments_output_path, sample_rate=target_sr)
        logger.info(f'已保存伴奏: {instruments_output_path} ({target_sr}Hz)')
        
        # Xóa file gốc để tiết kiệm dung lượng
        try:
            os.remove(audio_path)
            logger.info(f'Đã xóa file gốc để tiết kiệm dung lượng: {audio_path}')
        except Exception as e:
            logger.warning(f'Không thể xóa file gốc: {e}')

        return vocal_output_path, instruments_output_path

    except Exception as e:
        logger.error(f'Tách âm thanh thất bại: {str(e)}')
        # Có lỗi, giải phóng tài nguyên và ném ngoại lệ
        release_model()
        raise


def extract_audio_from_video(folder: str, video_path: str = None) -> bool:
    """
    Trích xuất âm thanh từ video. Nếu không có video_path, mặc định tìm download.mp4 trong thư mục.
    """
    if video_path is None:
        video_path = os.path.join(folder, 'download.mp4')
        
    if not os.path.exists(video_path):
        logger.error(f"Không tìm thấy video để trích xuất âm thanh: {video_path}")
        return False
        
    audio_path = os.path.join(folder, 'audio.wav')
    if os.path.exists(audio_path):
        logger.info(f'Đã trích xuất âm thanh: {folder}')
        return True
    logger.info(f'Đang trích xuất âm thanh từ video: {video_path} -> {audio_path}')

    os.system(
        f'ffmpeg -loglevel error -i "{video_path}" -vn -acodec pcm_s16le -ar 48000 -ac 2 "{audio_path}"')

    time.sleep(1)
    logger.info(f'Trích xuất âm thanh hoàn tất: {folder}')
    return True


def separate_all_audio_under_folder(root_folder: str, model_name: str = "htdemucs_ft", device: str = 'auto',
                                    progress: bool = True, shifts: int = 1, video_path: str = None) -> None:
    """
    Tách tất cả âm thanh trong thư mục.
    Nếu video_path được cung cấp, ưu tiên sử dụng nó cho thư mục gốc.
    """
    global separator
    vocal_output_path, instruments_output_path = None, None

    try:
        # 1. Trường hợp có video_path cụ thể cho thư mục này
        if video_path and os.path.exists(video_path):
            if 'audio_vocals.wav' not in os.listdir(root_folder):
                extract_audio_from_video(root_folder, video_path)
                vocal_output_path, instruments_output_path = separate_audio(root_folder, model_name, device, progress, shifts)
            else:
                vocal_output_path = os.path.join(root_folder, 'audio_vocals.wav')
                instruments_output_path = os.path.join(root_folder, 'audio_instruments.wav')
            return f'Tách âm thanh hoàn tất: {root_folder}', vocal_output_path, instruments_output_path

        # 2. Trường hợp duyệt folder (tương thích ngược)
        for subdir, dirs, files in os.walk(root_folder):
            if 'download.mp4' not in files and not video_path:
                continue
            
            # Ưu tiên dùng video_path nếu có, nếu không tìm download.mp4
            current_video = video_path if video_path else os.path.join(subdir, 'download.mp4')
            
            if 'audio.wav' not in files:
                extract_audio_from_video(subdir, current_video)
            if 'audio_vocals.wav' not in files:
                vocal_output_path, instruments_output_path = separate_audio(subdir, model_name, device, progress,
                                                                            shifts)
            elif 'audio_vocals.wav' in files and 'audio_instruments.wav' in files:
                vocal_output_path = os.path.join(subdir, 'audio_vocals.wav')
                instruments_output_path = os.path.join(subdir, 'audio_instruments.wav')
                logger.info(f'Đã tách âm thanh: {subdir}')

        logger.info(f'Đã hoàn tất tách tất cả âm thanh: {root_folder}')
        return f'Tất cả âm thanh đã được tách: {root_folder}', vocal_output_path, instruments_output_path

    except Exception as e:
        logger.error(f'Lỗi trong quá trình tách âm thanh: {str(e)}')
        # Có lỗi, giải phóng tài nguyên
        release_model()
        raise


if __name__ == '__main__':
    folder = r"outputs"
    separate_all_audio_under_folder(folder, shifts=0)