# -*- coding: utf-8 -*-
import json
import os
import subprocess
import time
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
    """Lấy thông số video: width, height, fps, duration."""
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height,r_frame_rate,duration', '-of', 'json', video_path]
    result = subprocess.run(command, capture_output=True, text=True)
    data = json.loads(result.stdout)['streams'][0]
    
    # Xử lý fps dạng phân số (ví dụ: "30/1")
    fps_parts = data['r_frame_rate'].split('/')
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
    
    return {
        'width': data['width'],
        'height': data['height'],
        'fps': fps,
        'duration': float(data.get('duration', 0))
    }


def convert_resolution(aspect_ratio, resolution='1080p', orig_w=None, orig_h=None):
    """Chuyển đổi độ phân giải mục tiêu dựa trên tỷ lệ khung hình."""
    if resolution is None or str(resolution).lower() in ['original', 'source', 'none']:
        if orig_w and orig_h:
            return orig_w, orig_h
        return 1920, 1080 # Fallback 1080p if orig not provided

    try:
        if isinstance(resolution, str) and resolution.endswith('p'):
            base_h = int(resolution[:-1])
        else:
            base_h = int(resolution)
            
        if aspect_ratio < 1:
            # Vertical/Portrait
            width = base_h
            height = int(width / aspect_ratio)
        else:
            # Horizontal/Landscape
            height = base_h
            width = int(height * aspect_ratio)
    except Exception as e:
        logger.warning(f"Resolution parse error: {e}. Fallback to 1080p.")
        return 1920, 1080

    # Đảm bảo chiều rộng và chiều cao chia hết cho 2
    width = width - width % 2
    height = height - height % 2
    
    return width, height
    
def synthesize_video(folder, subtitles=True, speed_up=1.00, fps=30, resolution='1080p', background_music=None, watermark_path=None, bgm_volume=0.5, video_volume=1.0, input_video=None):
    """Tổng hợp âm thanh, video và phụ đề hoàn chỉnh trong 1 pass duy nhất."""
    
    translation_path = os.path.join(folder, 'translation.json')
    input_audio = os.path.join(folder, 'audio_combined.wav')
    if input_video is None:
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
    target_w, target_h = convert_resolution(aspect_ratio, resolution, orig_w, orig_h)
    resolution_str = f'{target_w}x{target_h}'
    
    # Lấy thông số đồng bộ từ .env
    from dotenv import load_dotenv
    load_dotenv()
    MAX_PTS_FACTOR = float(os.getenv('MAX_PTS_FACTOR', 1.43))
    
    # Điều kiện tiên quyết cho Stream Copy (Siêu tốc):
    # Không watermark, không đổi phân giải, không đổi fps, không BGM, không phụ đề hardcode
    base_fast_conditions = (not watermark_path and 
                            speed_up == 1.0 and 
                            orig_w == target_w and 
                            orig_h == target_h and 
                            abs(video_info['fps'] - fps) < 0.1 and
                            not background_music)
    
    # Kiểm tra metadata đồng bộ thích ứng
    is_adaptive = all(k in translation[0] for k in ['original_start', 'original_end']) if translation else False
    
    # ƯU TIÊN: Nếu người dùng muốn giữ nguyên 1:1 (MAX_PTS_FACTOR=1.0) và đủ điều kiện copy
    if MAX_PTS_FACTOR == 1.0 and base_fast_conditions:
        logger.info("MAX_PTS_FACTOR=1.0 phát hiện. Sử dụng Stream Copy siêu tốc...")
        return run_fast_merge(input_video, input_audio, final_video)

    # TRƯỜNG HỢP 1: Đồng bộ thích ứng (từng câu)
    if is_adaptive:
        segments = []
        last_orig_end = 0.0
        last_target_end = 0.0
        
        for i, line in enumerate(translation):
            orig_start = line['original_start']
            orig_end = line['original_end']
            target_start = line['start']
            target_end = line['end']
            actual_dur = line.get('duration', target_end - target_start)
            
            # 1. Xử lý khoảng lặng
            if orig_start > last_orig_end:
                gap_orig_dur = orig_start - last_orig_end
                gap_target_dur = target_start - last_target_end
                if gap_orig_dur > 0:
                    pts = max(0.2, min(gap_target_dur / gap_orig_dur, MAX_PTS_FACTOR)) if gap_target_dur > 0 else 0.2
                    segments.append({'start': last_orig_end, 'end': orig_start, 'pts': pts, 'type': 'gap'})

            # 2. Xử lý giọng nói
            seg_orig_dur = orig_end - orig_start
            if seg_orig_dur > 0:
                pts = actual_dur / seg_orig_dur
                if pts > MAX_PTS_FACTOR:
                    logger.warning(f"Segment {i}: PTS {pts:.2f} vượt giới hạn. Cố định tại {MAX_PTS_FACTOR}.")
                    pts = MAX_PTS_FACTOR
                segments.append({'start': orig_start, 'end': orig_end, 'pts': pts, 'type': 'speech'})
            
            last_orig_end = orig_end
            last_target_end = target_end

        total_duration = video_info.get('duration', 0)
        if total_duration > last_orig_end:
            segments.append({'start': last_orig_end, 'end': total_duration, 'pts': 1.0, 'type': 'tail'})

        # LOG DURATION CHECK
        final_v_dur = sum([ (s['end'] - s['start']) * s['pts'] for s in segments ])
        logger.info(f"Dự kiến độ dài video: {final_v_dur:.2f}s (Gốc: {total_duration:.2f}s, Audio: {last_target_end:.2f}s)")

        # TỐI ƯU HÓA: Nếu tất cả PTS đều là 1.0 (khớp 1:1), dùng Stream Copy ngay lập tức
        if all(abs(s['pts'] - 1.0) < 0.01 for s in segments) and base_fast_conditions:
            logger.info("Phát hiện timeline khớp 1:1 hoàn hảo. Tự động chuyển sang Stream Copy...")
            return run_fast_merge(input_video, input_audio, final_video)

        # Build complex filter
        v_parts = []
        for i, seg in enumerate(segments):
            safe_start = min(seg['start'], total_duration - 0.1)
            safe_end = min(seg['end'], total_duration)
            v_parts.append(f"[0:v]trim=start={safe_start}:end={safe_end},setpts={seg['pts']}*(PTS-STARTPTS)[v{i}]")
        
        filter_complex = ";".join(v_parts) + ";" + "".join([f"[v{i}]" for i in range(len(v_parts))]) + f"concat=n={len(v_parts)}:v=1[v_synced]"
        v_map = "[v_synced]"

    # TRƯỜNG HỢP 2: Không đồng bộ thích ứng (chỉ có tốc độ tổng quát)
    else:
        if base_fast_conditions and speed_up == 1.0:
            logger.info("Tốc độ đồng nhất. Sử dụng Stream Copy siêu tốc...")
            return run_fast_merge(input_video, input_audio, final_video)
            
        v_filters = []
        if speed_up != 1.0:
            v_filters.append(f"setpts=PTS/{speed_up}")
        
        if v_filters:
            filter_complex = f"[0:v]{','.join(v_filters)}[v_processed]"
            v_map = "[v_processed]"
        else:
            filter_complex = ""
            v_map = "0:v"


    # --- Common Post-Processing Filters (Watermark, etc.) ---
    # These apply if we didn't return early from fast merge.
    
    input_args = ['-i', input_video, '-i', input_audio]
    
    # 3. Watermark
    if watermark_path and os.path.exists(watermark_path):
        input_args.extend(['-i', watermark_path])
        base_v = v_map
        filter_complex += f";[2:v]scale=iw*0.15:ih*0.15[wm];{base_v}[wm]overlay=W-w-10:H-h-10[v_out]"
        v_map = "[v_out]"

    ffmpeg_command = [
        'ffmpeg',
        *input_args,
        '-filter_complex', filter_complex,
        '-map', v_map,
        '-map', '1:a:0',
        '-r', str(fps),
        '-s', resolution_str,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-preset', 'veryfast',
        final_video,
        '-y',
        '-threads', '0',
        '-shortest',
    ]
    
    # 4. Background Music Mixing
    if background_music and os.path.exists(background_music):
        logger.info(f"Tối ưu hóa: Trộn nhạc nền trong cùng 1 pass FFmpeg từ: {background_music}")
        bgm_input_index = len(input_args) // 2
        input_args.extend(['-i', background_music])
        
        audio_filter = f"[1:a]volume={video_volume}[a_main];[{bgm_input_index}:a]volume={bgm_volume}[a_bgm];[a_main][a_bgm]amix=inputs=2:duration=first[a_out]"
        
        if filter_complex:
            filter_complex += f";{audio_filter}"
        else:
            filter_complex = audio_filter
            
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
            '-preset', 'veryfast',
            final_video,
            '-y',
            '-threads', '0',
        ]

    subprocess.run(ffmpeg_command)
    time.sleep(1)
    return final_video


def run_fast_merge(input_video, input_audio, output_video):
    """Thực hiện ghép video/audio bằng Stream Copy (không encode lại)."""
    ffmpeg_command = [
        'ffmpeg',
        '-i', input_video,
        '-i', input_audio,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        output_video,
        '-y'
    ]
    subprocess.run(ffmpeg_command)
    time.sleep(1)
    return output_video


def synthesize_all_video_under_folder(folder, subtitles=True, speed_up=1.00, fps=30, background_music=None, bgm_volume=0.5, video_volume=1.0, resolution='1080p', watermark_path="f_logo.png", original_video_path=None):
    """
    Duyệt qua các thư mục con và tổng hợp tất cả video tìm thấy.
    Nếu original_video_path được cung cấp, nó sẽ ưu tiên dùng video đó cho folder hiện tại.
    """
    watermark_path = None if not os.path.exists(watermark_path) else watermark_path
    output_video_path = None
    
    # 1. Trường hợp có video gốc cụ thể
    if original_video_path and os.path.exists(original_video_path):
        output_video_path = synthesize_video(folder, subtitles=subtitles,
                        speed_up=speed_up, fps=fps, resolution=resolution,
                        background_music=background_music,
                        watermark_path=watermark_path, bgm_volume=bgm_volume, video_volume=video_volume,
                        input_video=original_video_path)
        return f'Đã tổng hợp video: {folder}', output_video_path

    # 2. Trường hợp duyệt folder (tương thích ngược)
    for root, dirs, files in os.walk(folder):
        if 'download.mp4' in files:
            output_video_path = synthesize_video(root, subtitles=subtitles,
                            speed_up=speed_up, fps=fps, resolution=resolution,
                            background_music=background_music,
                            watermark_path=watermark_path, bgm_volume=bgm_volume, video_volume=video_volume)
            
    return f'Đã tổng hợp toàn bộ video trong {folder}', output_video_path


if __name__ == '__main__':
    print("Mô-đun tổng hợp video tối ưu đã sẵn sàng.")