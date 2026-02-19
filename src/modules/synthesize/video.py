# -*- coding: utf-8 -*-
import json
import os
import shutil
import string
import subprocess
import time
import random
import traceback

from loguru import logger


def split_text(input_data,
               punctuations=['，', '；', '：', '。', '？', '！', '\n', '”']):
    """
    Tách văn bản dựa trên các dấu câu kết thúc câu.
    Dành cho việc ngắt dòng phụ đề.
    """
    # Hàm kiểm tra xem một ký tự có phải là dấu câu kết thúc câu tiếng Trung hay không
    def is_punctuation(char):
        return char in punctuations

    # Xử lý từng mục trong dữ liệu đầu vào
    output_data = []
    for item in input_data:
        start = item["start"]
        text = item["translation"]
        speaker = item.get("speaker", "SPEAKER_00")
        original_text = item["text"]
        sentence_start = 0

        # Tính toán thời lượng cho mỗi ký tự
        duration_per_char = (item["end"] - item["start"]) / len(text)
        for i, char in enumerate(text):
            # Nếu ký tự là dấu câu, tách câu
            if not is_punctuation(char) and i != len(text) - 1:
                continue
            if i - sentence_start < 5 and i != len(text) - 1:
                continue
            if i < len(text) - 1 and is_punctuation(text[i+1]):
                continue
            sentence = text[sentence_start:i+1]
            sentence_end = start + duration_per_char * len(sentence)

            # Thêm mục mới vào danh sách đầu ra
            output_data.append({
                "start": round(start, 3),
                "end": round(sentence_end, 3),
                "text": original_text,
                "translation": sentence,
                "speaker": speaker
            })

            # Cập nhật thời điểm bắt đầu cho câu tiếp theo
            start = sentence_end
            sentence_start = i + 1

    return output_data
    
def format_timestamp(seconds):
    """Chuyển đổi giây sang định dạng thời gian SRT (HH:MM:SS,mmm)."""
    millisec = int((seconds - int(seconds)) * 1000)
    hours, seconds = divmod(int(seconds), 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millisec:03}"

def generate_srt(translation, srt_path, speed_up=1, max_line_char=30):
    """Tạo file phụ đề SRT từ dữ liệu dịch thuật."""
    translation = split_text(translation)
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, line in enumerate(translation):
            start = format_timestamp(line['start']/speed_up)
            end = format_timestamp(line['end']/speed_up)
            text = line['translation']
            line_count = len(text)//(max_line_char+1) + 1
            avg = min(round(len(text)/line_count), max_line_char)
            text = '\n'.join([text[j*avg:(j+1)*avg]
                             for j in range(line_count)])
            f.write(f'{i+1}\n')
            f.write(f'{start} --> {end}\n')
            f.write(f'{text}\n\n')


def get_video_info(video_path):
    """Lấy thông số video: width, height, fps."""
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height,r_frame_rate', '-of', 'json', video_path]
    result = subprocess.run(command, capture_output=True, text=True)
    data = json.loads(result.stdout)['streams'][0]
    
    # Xử lý fps dạng phân số (ví dụ: "30/1")
    fps_parts = data['r_frame_rate'].split('/')
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
    
    return {
        'width': data['width'],
        'height': data['height'],
        'fps': fps
    }


def convert_resolution(aspect_ratio, resolution='1080p'):
    """Chuyển đổi độ phân giải mục tiêu dựa trên tỷ lệ khung hình."""
    if aspect_ratio < 1:
        width = int(resolution[:-1])
        height = int(width / aspect_ratio)
    else:
        height = int(resolution[:-1])
        width = int(height * aspect_ratio)
    # Đảm bảo chiều rộng và chiều cao chia hết cho 2
    width = width - width % 2
    height = height - height % 2
    
    return width, height
    
def synthesize_video(folder, subtitles=True, speed_up=1.00, fps=30, resolution='1080p', background_music=None, watermark_path=None, bgm_volume=0.5, video_volume=1.0):
    """Tổng hợp âm thanh, video và phụ đề hoàn chỉnh trong 1 pass duy nhất."""
    
    translation_path = os.path.join(folder, 'translation.json')
    input_audio = os.path.join(folder, 'audio_combined.wav')
    input_video = os.path.join(folder, 'download.mp4')
    
    if not os.path.exists(translation_path) or not os.path.exists(input_audio):
        logger.warning(f"Thiếu file translation.json hoặc audio_combined.wav tại {folder}")
        return
    
    with open(translation_path, 'r', encoding='utf-8') as f:
        translation = json.load(f)
        
    srt_path = os.path.join(folder, 'subtitles.srt')
    final_video = os.path.join(folder, 'video.mp4')
    
    # Tạo file phụ đề SRT (luôn tạo file rời để người dùng sử dụng)
    generate_srt(translation, srt_path, speed_up)
    # Đường dẫn SRT cần thay đổi để FFmpeg subtitles filter nhận diện đúng trên Windows/Linux
    srt_path_ffmpeg = srt_path.replace('\\', '/').replace(':', '\\:')
    
    # Lấy thông số video gốc
    video_info = get_video_info(input_video)
    orig_w, orig_h = video_info['width'], video_info['height']
    aspect_ratio = orig_w / orig_h
    
    # Độ phân giải đích
    target_w, target_h = convert_resolution(aspect_ratio, resolution)
    resolution_str = f'{target_w}x{target_h}'
    
    # Kiểm tra xem có thể dùng Stream Copy không?
    # Điều kiện: Không phụ đề, không watermark, không đổi speed, không đổi độ phân giải/fps
    can_stream_copy = (not subtitles and 
                       not watermark_path and 
                       speed_up == 1.0 and 
                       orig_w == target_w and 
                       orig_h == target_h and 
                       abs(video_info['fps'] - fps) < 0.1)

    if can_stream_copy:
        logger.info("Phát hiện các thông số khớp nhau. Sử dụng Stream Copy để tổng hợp siêu tốc...")
        ffmpeg_command = [
            'ffmpeg',
            '-i', input_video,
            '-i', input_audio,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            final_video,
            '-y'
        ]
    else:
        logger.info("Bắt đầu tổng hợp video với Encoding đơn luồng (Single-pass)...")
        # Video filters
        v_filters = []
        if speed_up != 1.0:
            v_filters.append(f"setpts=PTS/{speed_up}")
            
        # Hardcode phụ đề nếu bật
        if subtitles:
            font_size = int(target_w/128)
            outline = int(round(font_size/8))
            style = f"FontName=SimHei,FontSize={font_size},PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline={outline},WrapStyle=2"
            v_filters.append(f"subtitles='{srt_path_ffmpeg}':force_style='{style}'")

        # Watermark
        input_args = ['-i', input_video, '-i', input_audio]
        filter_complex = ""
        
        if watermark_path and os.path.exists(watermark_path):
            input_args.extend(['-i', watermark_path])
            base_v = "[0:v]"
            if v_filters:
                filter_complex += f"[0:v]{','.join(v_filters)}[v_processed];"
                base_v = "[v_processed]"
            
            filter_complex += f"[2:v]scale=iw*0.15:ih*0.15[wm];{base_v}[wm]overlay=W-w-10:H-h-10[v_out]"
            v_map = "[v_out]"
        else:
            if v_filters:
                filter_complex += f"[0:v]{','.join(v_filters)}[v_out]"
                v_map = "[v_out]"
            else:
                v_map = "0:v"

        # Audio filter
        a_filter = f"[1:a]atempo={speed_up}[a_out]"
        if filter_complex:
            filter_complex += f";{a_filter}"
        else:
            filter_complex = a_filter
        
        ffmpeg_command = [
            'ffmpeg',
            *input_args,
            '-filter_complex', filter_complex,
            '-map', v_map,
            '-map', '[a_out]',
            '-r', str(fps),
            '-s', resolution_str,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-preset', 'veryfast', # Tăng tốc độ encode
            final_video,
            '-y',
            '-threads', '0', # Sử dụng toàn bộ core CPU
        ]

    subprocess.run(ffmpeg_command)
    time.sleep(1)

    # Thêm nhạc nền (Single-pass master mixing)
    if background_music and os.path.exists(background_music):
        logger.info(f"Đang trộn nhạc nền từ: {background_music}")
        final_video_with_bgm = final_video.replace('.mp4', '_bgm.mp4')
        ffmpeg_command_bgm = [
            'ffmpeg',
            '-i', final_video,
            '-i', background_music,
            '-filter_complex', f'[0:a]volume={video_volume}[v0];[1:a]volume={bgm_volume}[v1];[v0][v1]amix=inputs=2:duration=first[a]',
            '-map', '0:v',
            '-map', '[a]',
            '-c:v', 'copy', # Copy video stream, tránh re-encode lần 2
            '-c:a', 'aac',
            final_video_with_bgm,
            '-y'
        ]
        subprocess.run(ffmpeg_command_bgm)
        os.remove(final_video)
        os.rename(final_video_with_bgm, final_video)
        time.sleep(1)

    return final_video


def synthesize_all_video_under_folder(folder, subtitles=True, speed_up=1.00, fps=30, background_music=None, bgm_volume=0.5, video_volume=1.0, resolution='1080p', watermark_path="f_logo.png"):
    """Duyệt qua các thư mục con và tổng hợp tất cả video tìm thấy."""
    watermark_path = None if not os.path.exists(watermark_path) else watermark_path
    output_video_path = None
    for root, dirs, files in os.walk(folder):
        if 'download.mp4' in files:
            output_video_path = synthesize_video(root, subtitles=subtitles,
                            speed_up=speed_up, fps=fps, resolution=resolution,
                            background_music=background_music,
                            watermark_path=watermark_path, bgm_volume=bgm_volume, video_volume=video_volume)
            
    return f'Đã tổng hợp toàn bộ video trong {folder}', output_video_path


if __name__ == '__main__':
    print("Mô-đun tổng hợp video tối ưu đã sẵn sàng.")