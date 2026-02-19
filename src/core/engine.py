import json
import os
import time
import traceback

import torch
from loguru import logger
from src.utils.separation import separate_all_audio_under_folder, release_model
from src.modules.asr.manager import transcribe_all_audio_under_folder
from src.modules.asr.whisperx import init_whisperx, init_diarize, release_whisperx
from src.modules.translation.manager import translate_all_transcript_under_folder
from src.modules.tts.manager import generate_all_wavs_under_folder, init_TTS
from src.modules.synthesize.video import synthesize_all_video_under_folder
from src.utils.perf import PerformanceTracker

models_initialized = {'separator': False, 'whisperx': False, 'diarize': False}

def get_available_gpu_memory():
    try:
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            return free_memory / (1024 ** 3)
        return 0
    except: return 0

def initialize_models(tts_method, diarization):
    logger.info("Lazy loading models.")
    return

def process_video(video_file, root_folder,
                  separator_model, device, shifts,
                  whisper_model, batch_size, diarization, whisper_min_speakers, whisper_max_speakers,
                  translation_method, translation_target_language,
                  tts_method, tts_target_language, voice,
                  subtitles, speed_up, fps, background_music, bgm_volume, video_volume,
                  target_resolution, max_retries, progress_callback=None, tracker=None, audio_only=False, language=None,
                  asr_method='google', google_key=None):
    stages = [("Tải video...", 10), ("Tách tiếng người...", 15), ("Nhận diện giọng nói AI...", 20), ("Dịch phụ đề...", 25), ("Tổng hợp giọng nói AI...", 20), ("Tổng hợp video...", 10)]
    current_stage = 0
    progress_base = 0
    if progress_callback: progress_callback(0, "Đang chuẩn bị...")

    for retry in range(max_retries):
        current_stage = 0
        progress_base = 0
        try:
            stage_name, stage_weight = stages[current_stage+1]
            if progress_callback: progress_callback(progress_base + stages[0][1], stage_name)
            if tracker: tracker.start_stage("Preparation")
            base_name = os.path.splitext(os.path.basename(video_file))[0]
            import re
            base_name = re.sub(r'[\\/*?:"<>|]', '', base_name)
            folder = os.path.join(root_folder, base_name)
            if not os.path.exists(folder): os.makedirs(folder)
            
            current_stage += 1
            progress_base += stage_weight
            stage_name, stage_weight = stages[current_stage]
            if progress_callback: progress_callback(progress_base, stage_name)
            if tracker: tracker.end_stage("Preparation")
            
            if tracker: tracker.start_stage("Separation")
            status, vocals_path, _ = separate_all_audio_under_folder(folder, model_name=separator_model, video_path=video_file)
            if not vocals_path or not os.path.exists(vocals_path): raise Exception(f"Vocal file missing: {vocals_path}")
            if os.getenv('KEEP_MODELS') != 'True': release_model(); torch.cuda.empty_cache()
            if tracker: tracker.end_stage("Separation")

            current_stage += 1
            progress_base += stage_weight
            stage_name, stage_weight = stages[current_stage]
            if progress_callback: progress_callback(progress_base, stage_name)
            if tracker: tracker.start_stage("ASR")
            status, result_json = transcribe_all_audio_under_folder(folder, whisper_model_name=whisper_model, device=device, batch_size=batch_size, diarization=diarization, language=language, asr_method=asr_method, google_key=google_key)
            if not result_json: raise Exception("ASR result empty")
            if tracker: tracker.end_stage("ASR")
            if os.getenv('KEEP_MODELS') != 'True': release_whisperx(); torch.cuda.empty_cache()

            current_stage += 1
            progress_base += stage_weight
            stage_name, stage_weight = stages[current_stage]
            if progress_callback: progress_callback(progress_base, stage_name)
            if tracker: tracker.start_stage("Translation")
            status, summary, translation = translate_all_transcript_under_folder(folder, method=translation_method, target_language=translation_target_language)
            if not translation: raise Exception("Translation empty")
            if tracker: tracker.end_stage("Translation")

            current_stage += 1
            progress_base += stage_weight
            stage_name, stage_weight = stages[current_stage]
            if progress_callback: progress_callback(progress_base, stage_name)
            if tracker: tracker.start_stage("TTS")
            status, synth_path, _ = generate_all_wavs_under_folder(folder, method=tts_method, target_language=tts_target_language, voice=voice, video_volume=video_volume)
            if not synth_path or not os.path.exists(synth_path): raise Exception("TTS file missing")
            if tracker: tracker.end_stage("TTS")

            if not audio_only:
                current_stage += 1
                progress_base += stage_weight
                stage_name, stage_weight = stages[current_stage]
                if progress_callback: progress_callback(progress_base, stage_name)
                if tracker: tracker.start_stage("Synthesis")
                status, output_video = synthesize_all_video_under_folder(folder, subtitles=subtitles, speed_up=speed_up, fps=fps, resolution=target_resolution, background_music=background_music, bgm_volume=bgm_volume, video_volume=video_volume, original_video_path=video_file)
                if not output_video or not os.path.exists(output_video): raise Exception("Synthesis failed")
                if tracker: tracker.end_stage("Synthesis")
            else: output_video = os.path.join(folder, "audio_combined.wav")
            
            if progress_callback: progress_callback(100, "Xử lý thành công!")
            if tracker: tracker.finalize(); tracker.save_stats(os.path.join(folder, "timing_stats.json"))
            return True, output_video, "Xử lý thành công"
        except Exception as e:
            logger.exception("An error occurred during processing:")
            if retry < max_retries - 1: logger.info(f"Retry {retry+2}...")
            else: return False, None, f"Lỗi: {str(e)}"
    return False, None, "Retry limit reached"

def engine_run(root_folder='outputs', url=None, video_file=None, num_videos=1,
                  separator_model='UVR-MDX-NET-Inst_HQ_3.onnx', device='auto', shifts=1,
                  whisper_model='small', batch_size=4, diarization=False,
                  whisper_min_speakers=None, whisper_max_speakers=None,
                  translation_method='LLM', translation_target_language='vi',
                  tts_method='auto', tts_target_language='vi', voice=None,
                  subtitles=False, speed_up=1.00, fps=30,
                  background_music=None, bgm_volume=0.5, video_volume=1.0, target_resolution='1080p',
                  max_retries=5, progress_callback=None, audio_only=False, language=None,
                  asr_method='google', google_key=None):
    try:
        tracker = PerformanceTracker()
        tracker.start_stage("Model Initialization")
        initialize_models(tts_method, diarization)
        tracker.end_stage("Model Initialization")
        if video_file:
            success, output_video, error_msg = process_video(video_file, root_folder, separator_model, device, shifts, whisper_model, batch_size, diarization, whisper_min_speakers, whisper_max_speakers, translation_method, translation_target_language, tts_method, tts_target_language, voice, subtitles, speed_up, fps, background_music, bgm_volume, video_volume, target_resolution, max_retries, progress_callback, tracker, audio_only=audio_only, language=language, asr_method=asr_method, google_key=google_key)
            if success: return 'Xử lý thành công', output_video
            else: return f'Xử lý thất bại: {error_msg}', None
        return "Lỗi: File video không tồn tại", None
    except Exception as e: return f"Lỗi: {str(e)}", None

if __name__ == '__main__':
    engine_run(root_folder='outputs', video_file='test.mp4', translation_method='LLM')
