import json
import os
import time
import traceback
import shutil
import hashlib

import torch
from loguru import logger
from utils.separation import separate_all_audio_under_folder, init_demucs, release_model
from utils.dereverb import process_folder_dereverb, init_dereverb, release_dereverb
from modules.asr.manager import transcribe_all_audio_under_folder
from modules.asr.whisperx import init_whisperx, init_diarize, release_whisperx
from modules.asr.funasr import init_funasr
from modules.translation.manager import translate_all_transcript_under_folder
from modules.tts.manager import generate_all_wavs_under_folder, init_TTS
from modules.synthesize.video import synthesize_all_video_under_folder
from utils.utils import save_wav
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.perf import PerformanceTracker

# Theo dõi trạng thái khởi tạo mô hình
models_initialized = {
    'demucs': False,
    'whisperx': False,
    'diarize': False,
    'funasr': False,
    'dereverb': False
}


def get_available_gpu_memory():
    """Lấy dung lượng bộ nhớ GPU hiện có (GB)"""
    try:
        if torch.cuda.is_available():
            # Lấy dung lượng bộ nhớ trống trên thiết bị hiện tại
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            return free_memory / (1024 ** 3)  # Chuyển đổi sang GB
        return 0  # Nếu không có GPU hoặc CUDA không khả dụng
    except Exception:
        return 0  # Trả về 0 nếu có lỗi


def initialize_models(tts_method, asr_method, diarization):
    """
    Khởi tạo các mô hình cần thiết.
    Chỉ khởi tạo mô hình trong lần gọi đầu tiên để tránh tải lại.
    """
    # Sử dụng trạng thái toàn cục để theo dõi các mô hình đã khởi tạo
    global models_initialized

    with ThreadPoolExecutor() as executor:
        try:
            # Khởi tạo mô hình Demucs
            if not models_initialized['demucs']:
                executor.submit(init_demucs)
                models_initialized['demucs'] = True
                logger.info("Khởi tạo mô hình Demucs hoàn tất")
            
            # Khởi tạo mô hình Dereverb
            if not models_initialized.get('dereverb', False):
                executor.submit(init_dereverb)
                models_initialized['dereverb'] = True
                logger.info("Đang khởi tạo mô hình khử vang DeepFilterNet...")

            # LƯU Ý: Không khởi tạo WhisperX ở đây để tránh OOM (std::bad_alloc)
            # Nó sẽ được tải lười (lazy load) khi thực sự cần thiết trong pipeline
            if asr_method == 'FunASR' and not models_initialized['funasr']:
                executor.submit(init_funasr)
                models_initialized['funasr'] = True
                logger.info("Khởi tạo mô hình FunASR hoàn tất")

        except Exception as e:
            stack_trace = traceback.format_exc()
            logger.error(f"Khởi tạo mô hình thất bại: {str(e)}\n{stack_trace}")
            # Reset trạng thái khởi tạo nếu có lỗi
            models_initialized = {key: False for key in models_initialized}
            release_model()  # Giải phóng các mô hình đã tải
            raise


def process_video(video_file, root_folder, resolution,
                  demucs_model, device, shifts,
                  asr_method, whisper_model, batch_size, diarization, whisper_min_speakers, whisper_max_speakers,
                  translation_method, translation_target_language,
                  tts_method, tts_target_language, voice,
                  subtitles, speed_up, fps, background_music, bgm_volume, video_volume,
                  target_resolution, max_retries, progress_callback=None, tracker=None):
    """
    Quy trình xử lý hoàn chỉnh cho một video duy nhất với hỗ trợ callback tiến độ.

    Args:
        progress_callback: Hàm callback để báo cáo tiến độ và trạng thái, định dạng progress_callback(percent, message)
    """
    local_time = time.localtime()

    # Định nghĩa các giai đoạn và trọng số tiến độ
    stages = [
        ("Tải video...", 10),  # 10%
        ("Tách tiếng người...", 15),  # 15%
        ("Nhận diện giọng nói AI...", 20),  # 20%
        ("Dịch phụ đề...", 25),  # 25%
        ("Tổng hợp giọng nói AI...", 20),  # 20%
        ("Tổng hợp video...", 10)  # 10%
    ]

    current_stage = 0
    progress_base = 0

    # Báo cáo tiến độ ban đầu
    if progress_callback:
        progress_callback(0, "Đang chuẩn bị xử lý...")

    for retry in range(max_retries):
        try:
            # Chuyển sang giai đoạn tách tiếng (Bỏ qua giai đoạn download vì dùng file cục bộ)
            stage_name, stage_weight = stages[current_stage+1]
            if progress_callback:
                progress_callback(progress_base + stages[0][1], stage_name)
            
            if tracker: tracker.start_stage("Preparation")
            
            # Sử dụng root_folder/vidhash làm thư mục xử lý
            # Sử dụng tên video gốc để tạo thư mục dự án (giúp gom nhóm chuyên nghiệp)
            base_name = os.path.splitext(os.path.basename(video_file))[0]
            # Strip UUID prefix if present (e.g., 6db1bbbe-8c06-426a-9174-0ec4e217579d_video3.mp4)
            if len(base_name) > 37 and base_name[36] == '_':
                try:
                    # Check if the first 36 chars look like a UUID
                    import uuid
                    uuid.UUID(base_name[:36])
                    base_name = base_name[37:]
                except ValueError:
                    pass
            
            # Loại bỏ ký tự đặc biệt để an toàn cho đường dẫn
            import re
            base_name = re.sub(r'[\\/*?:"<>|]', '', base_name)
            folder = os.path.join(root_folder, base_name)
            
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            logger.info(f'Đang xử lý video từ nguồn: {video_file} (Thư mục làm việc: {folder})')

            # Cập nhật tiến độ sau khi chuẩn bị xong file
            current_stage += 1
            progress_base += stage_weight
            stage_name, stage_weight = stages[current_stage]
            if progress_callback:
                progress_callback(progress_base, stage_name)

            if tracker: tracker.end_stage("Preparation")
            if tracker: tracker.start_stage("Separation")

            try:
                # Chuyền video_file gốc vào để tránh copy
                status, vocals_path, _ = separate_all_audio_under_folder(
                    folder, model_name=demucs_model, device=device, progress=True, shifts=shifts, video_path=video_file)
                logger.info(f'Tách tiếng hoàn tất: {vocals_path}')
                
                # Giải phóng VRAM của Demucs
                release_model()
                torch.cuda.empty_cache()
                
                if tracker: tracker.end_stage("Separation")
                
                # Giai đoạn mới: Khử vang (Studio-Grade Upgrade)
                if progress_callback:
                    progress_callback(progress_base + 5, "Đang thực hiện khử vang tiếng người (Dereverb)...")
                
                if tracker: tracker.start_stage("Dereverb")
                dereverb_path = process_folder_dereverb(folder)
                if dereverb_path:
                    logger.info(f'Khử vang hoàn tất: {dereverb_path}')
                else:
                    logger.warning('Bỏ qua khử vang (File không tồn tại hoặc lỗi)')
                
                if tracker: tracker.end_stage("Dereverb")
                
                # Giải phóng VRAM của Dereverb
                release_dereverb()
                torch.cuda.empty_cache()
            except Exception as e:
                stack_trace = traceback.format_exc()
                error_msg = f'Tách tiếng thất bại: {str(e)}\n{stack_trace}'
                logger.error(error_msg)
                return False, None, error_msg

            # Chuyển sang giai đoạn ASR
            current_stage += 1
            progress_base += stage_weight
            stage_name, stage_weight = stages[current_stage]
            if progress_callback:
                progress_callback(progress_base, stage_name)

            if tracker: tracker.start_stage("ASR")

            try:
                status, result_json = transcribe_all_audio_under_folder(
                    folder, asr_method=asr_method, whisper_model_name=whisper_model, device=device,
                    batch_size=batch_size, diarization=diarization,
                    min_speakers=whisper_min_speakers,
                    max_speakers=whisper_max_speakers)
                logger.info(f'Nhận diện giọng nói hoàn tất: {status}')
                
                if tracker: tracker.end_stage("ASR")
                
                # Giải phóng VRAM của WhisperX
                release_whisperx()
                torch.cuda.empty_cache()
            except Exception as e:
                stack_trace = traceback.format_exc()
                error_msg = f'Nhận diện giọng nói thất bại: {str(e)}\n{stack_trace}'
                logger.error(error_msg)
                return False, None, error_msg

            # Chuyển sang giai đoạn dịch thuật
            current_stage += 1
            progress_base += stage_weight
            stage_name, stage_weight = stages[current_stage]
            if progress_callback:
                progress_callback(progress_base, stage_name)

            if tracker: tracker.start_stage("Translation")

            try:
                status, summary, translation = translate_all_transcript_under_folder(
                    folder, method=translation_method, target_language=translation_target_language)
                logger.info(f'Dịch thuật hoàn tất: {status}')
                if tracker: tracker.end_stage("Translation")
            except Exception as e:
                stack_trace = traceback.format_exc()
                error_msg = f'Dịch thuật thất bại: {str(e)}\n{stack_trace}'
                logger.error(error_msg)
                return False, None, error_msg

            # Chuyển sang giai đoạn TTS
            current_stage += 1
            progress_base += stage_weight
            stage_name, stage_weight = stages[current_stage]
            if progress_callback:
                progress_callback(progress_base, stage_name)

            if tracker: tracker.start_stage("TTS")

            try:
                status, synth_path, _ = generate_all_wavs_under_folder(
                    folder, method=tts_method, target_language=tts_target_language, voice=voice, video_volume=video_volume)
                logger.info(f'Tổng hợp giọng nói hoàn tất: {synth_path}')
                if tracker: tracker.end_stage("TTS")
            except Exception as e:
                stack_trace = traceback.format_exc()
                error_msg = f'Tổng hợp giọng nói thất bại: {str(e)}\n{stack_trace}'
                logger.error(error_msg)
                return False, None, error_msg

            # Chuyển sang giai đoạn tổng hợp video
            current_stage += 1
            progress_base += stage_weight
            stage_name, stage_weight = stages[current_stage]
            if progress_callback:
                progress_callback(progress_base, stage_name)

            if tracker: tracker.start_stage("Synthesis")

            try:
                status, output_video = synthesize_all_video_under_folder(
                    folder, subtitles=subtitles, speed_up=speed_up, fps=fps, resolution=target_resolution,
                    background_music=background_music, bgm_volume=bgm_volume, video_volume=video_volume, original_video_path=video_file)
                logger.info(f'Tổng hợp video hoàn tất: {output_video}')
                if tracker: tracker.end_stage("Synthesis")
            except Exception as e:
                stack_trace = traceback.format_exc()
                error_msg = f'Tổng hợp video thất bại: {str(e)}\n{stack_trace}'
                logger.error(error_msg)
                return False, None, error_msg

            # Hoàn tất tất cả các giai đoạn
            if progress_callback:
                progress_callback(100, "Xử lý hoàn tất!")

            if tracker:
                tracker.finalize()
                tracker.save_stats(os.path.join(folder, "timing_stats.json"))
            
            # Dọn dẹp siêu sạch (Ultra-Clean Cleanup) sau khi thành công
            logger.info("Đang thực hiện dọn dẹp các tệp trung gian... (Tạm vô hiệu hóa để debug)")
            # clean_files = [
            #     'audio_vocals.wav', 'audio_vocals_dereverb.wav',
            #     'audio_instruments.wav', 'audio_tts.wav', 'audio_combined.wav',
            #     'transcript.json', 'summary.json', 'timing_stats.json'
            # ]
            # for f in clean_files:
            #     fpath = os.path.join(folder, f)
            #     if os.path.exists(fpath):
            #         os.remove(fpath)
            
            # Xóa thư mục SPEAKER nếu có
            # speaker_dir = os.path.join(folder, 'SPEAKER')
            # if os.path.exists(speaker_dir):
            #     shutil.rmtree(speaker_dir)
            
            logger.info("Dọn dẹp hoàn tất. Chỉ giữ lại Video, Subtitles và Translation JSON.")
                
            return True, output_video, "Xử lý thành công"
        except Exception as e:
            stack_trace = traceback.format_exc()
            error_msg = f'Lỗi khi xử lý video: {str(e)}\n{stack_trace}'
            logger.error(error_msg)
            if retry < max_retries - 1:
                logger.info(f'Đang thử lại {retry + 2}/{max_retries}...')
            else:
                return False, None, error_msg

    return False, None, f"Đã đạt số lần thử lại tối đa: {max_retries}"


def engine_run(root_folder='outputs', url=None, video_file=None, num_videos=1, resolution='1080p',
                  demucs_model='htdemucs_ft', device='auto', shifts=1,
                  asr_method='WhisperX', whisper_model='large', batch_size=32, diarization=False,
                  whisper_min_speakers=None, whisper_max_speakers=None,
                  translation_method='LLM', translation_target_language='简体中文',
                  tts_method='auto', tts_target_language='简体中文', voice=None,
                  subtitles=False, speed_up=1.00, fps=30,
                  background_music=None, bgm_volume=0.5, video_volume=1.0, target_resolution='1080p',
                  max_workers=3, max_retries=5, progress_callback=None):
    """
    Hàm chạy chính toàn bộ quy trình xử lý video.

    Args:
        progress_callback: Hàm callback để báo cáo tiến độ và trạng thái.
    """
    try:
        success_list = []
        fail_list = []
        error_details = []

        # Ghi nhật ký bắt đầu nhiệm vụ và tất cả các thông số
        logger.info("-" * 50)
        logger.info(f"Bắt đầu nhiệm vụ xử lý: {video_file if video_file else url}")
        logger.info(f"Thông số: Thư mục gốc={root_folder}, Độ phân giải={resolution}")
        logger.info(f"Tách tiếng: Mô hình={demucs_model}, Thiết bị={device}, Số lần dịch chuyển={shifts}")
        logger.info(f"Nhận diện: Phương pháp={asr_method}, Mô hình={whisper_model}, Batch Size={batch_size}")
        logger.info(f"Dịch thuật: Phương pháp={translation_method}, Ngôn ngữ mục tiêu={translation_target_language}")
        logger.info(f"TTS: Phương pháp={tts_method}, Ngôn ngữ={tts_target_language}, Giọng={voice}")
        logger.info(f"Tổng hợp: Phụ đề={subtitles}, Tăng tốc={speed_up}, FPS={fps}, Phân giải đầu ra={target_resolution}")
        logger.info("-" * 50)

        # Khởi tạo mô hình
        try:
            if progress_callback:
                progress_callback(5, "Đang khởi tạo các mô hình AI...")
            
            tracker = PerformanceTracker()
            tracker.start_stage("Model Initialization")
            initialize_models(tts_method, asr_method, diarization)
            tracker.end_stage("Model Initialization")
        except Exception as e:
            stack_trace = traceback.format_exc()
            logger.error(f"Khởi tạo mô hình thất bại: {str(e)}\n{stack_trace}")
            return f"Khởi tạo mô hình thất bại: {str(e)}", None

        if video_file:
            success, output_video, error_msg = process_video(
                video_file, root_folder, resolution,
                demucs_model, device, shifts,
                asr_method, whisper_model, batch_size, diarization, whisper_min_speakers, whisper_max_speakers,
                translation_method, translation_target_language,
                tts_method, tts_target_language, voice,
                subtitles, speed_up, fps, background_music, bgm_volume, video_volume,
                target_resolution, max_retries, progress_callback, tracker
            )

            if success:
                logger.info(f"Xử lý video thành công: {video_file}")
                return 'Xử lý thành công', output_video
            else:
                logger.error(f"Xử lý video thất bại: {video_file}, Lỗi: {error_msg}")
                return f'Xử lý thất bại: {error_msg}', None
        else:
            return "Lỗi: Không cung cấp tệp video hợp lệ", None

    except Exception as e:
        # Bắt bất kỳ lỗi nào trong quá trình xử lý tổng thể
        stack_trace = traceback.format_exc()
        error_msg = f"Đã xảy ra lỗi trong quá trình thực thi: {str(e)}\n{stack_trace}"
        logger.error(error_msg)
        return error_msg, None


if __name__ == '__main__':
    engine_run(
        root_folder='outputs',
        video_file='test.mp4',
        translation_method='LLM'
    )