import os
import torch
import torchaudio
from df.enhance import enhance, init_df, load_audio, save_audio
from loguru import logger
import time

# Biến toàn cục cho trạng thái mô hình
df_model = None
df_state = None

def init_dereverb():
    """
    Khởi tạo mô hình DeepFilterNet.
    """
    global df_model, df_state
    if df_model is None:
        logger.info("Đang khởi tạo DeepFilterNet để khử vang...")
        t_start = time.time()
        # init_df sẽ tải model về ~/.cache/deepfilternet nếu chưa có
        df_model, df_state, _ = init_df()
        t_end = time.time()
        logger.info(f"DeepFilterNet đã được khởi tạo trong {t_end - t_start:.2f}s")

def release_dereverb():
    """
    Giải phóng tài nguyên mô hình DeepFilterNet.
    """
    global df_model, df_state
    if df_model is not None:
        logger.info("Đang giải phóng mô hình DeepFilterNet...")
        df_model = None
        df_state = None
        torch.cuda.empty_cache()
        import gc
        gc.collect()

def dereverb_audio(input_path: str, output_path: str = None):
    """
    Áp dụng khử vang cho file âm thanh bằng DeepFilterNet.
    """
    global df_model, df_state
    
    if output_path is None:
        output_path = input_path.replace('.wav', '_dereverb.wav')
        
    if os.path.exists(output_path):
        logger.info(f"Audio đã khử vang tồn tại: {output_path}")
        return output_path

    if df_model is None:
        init_dereverb()

    logger.info(f"Đang khử vang cho {input_path}...")
    t_start = time.time()
    try:
        # DeepFilterNet mặc định mong đợi 48kHz cho các model mới nhất
        audio, _ = load_audio(input_path, sr=df_state.sr())
        enhanced = enhance(df_model, df_state, audio)
        save_audio(output_path, enhanced, df_state.sr())
        t_end = time.time()
        logger.info(f"Khử vang hoàn tất trong {t_end - t_start:.2f}s. Đã lưu tại {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Khử vang thất bại: {e}")
        return input_path # Fallback về file gốc

def process_folder_dereverb(folder: str):
    """
    Khử vang phần vocal trong một thư mục cụ thể.
    """
    vocal_path = os.path.join(folder, 'audio_vocals.wav')
    if os.path.exists(vocal_path):
        dereverb_path = os.path.join(folder, 'audio_vocals_dereverb.wav')
        return dereverb_audio(vocal_path, dereverb_path)
    return None

if __name__ == '__main__':
    # Test trên một thư mục cụ thể nếu cần
    test_folder = "compare_results/temp/video3"
    if os.path.exists(test_folder):
        process_folder_dereverb(test_folder)
