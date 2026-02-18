import gradio as gr
from tools.utils.step000_video_downloader import download_from_url
from tools.utils.step010_demucs_vr import separate_all_audio_under_folder
from tools.asr.step020_asr import transcribe_all_audio_under_folder
from tools.translation.step030_translation import translate_all_transcript_under_folder
from tools.tts.step040_tts import generate_all_wavs_under_folder
from tools.synthesize.step050_synthesize_video import synthesize_all_video_under_folder
from tools.do_everything import do_everything
from tools.utils.utils import SUPPORT_VOICE
from tools.utils.i18n import i18n

def create_full_auto_interface():
    with gr.Blocks() as interface:
        with gr.Row():
            with gr.Column():
                video_file = gr.Video(label=i18n.get('select_local_video'))
                video_url = gr.Textbox(label=i18n.get('video_url'), placeholder=i18n.get('video_url_placeholder'), 
                           value='https://www.bilibili.com/video/BV1kr421M7vz/')
                video_output_folder = gr.Textbox(label=i18n.get('video_output_folder'), value='videos')
                download_video_count = gr.Slider(minimum=1, maximum=100, step=1, label=i18n.get('download_video_count'), value=5)
                resolution = gr.Radio(['4320p', '2160p', '1440p', '1080p', '720p', '480p', '360p', '240p', '144p'], label=i18n.get('resolution'), value='1080p')

                model = gr.Radio(['htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'hdemucs_mmi', 'mdx', 'mdx_extra', 'mdx_q', 'mdx_extra_q', 'SIG'], label=i18n.get('model'), value='htdemucs_ft')
                compute_device = gr.Radio(['auto', 'cuda', 'cpu'], label=i18n.get('compute_device'), value='auto')
                number_of_shifts = gr.Slider(minimum=0, maximum=10, step=1, label=i18n.get('number_of_shifts'), value=5)

                asr_model_selection = gr.Dropdown(['WhisperX', 'FunASR'], label=i18n.get('asr_model_selection'), value='WhisperX')
                whisperx_model_size = gr.Radio(['large', 'medium', 'small', 'base', 'tiny'], label=i18n.get('whisperx_model_size'), value='base')
                batch_size = gr.Slider(minimum=1, maximum=128, step=1, label=i18n.get('batch_size'), value=32)
                separate_speakers = gr.Checkbox(label=i18n.get('separate_speakers'), value=True)
                min_speakers = gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label=i18n.get('min_speakers'), value=None)
                max_speakers = gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label=i18n.get('max_speakers'), value=None)

                translation_method = gr.Dropdown(['OpenAI', 'LLM', 'Google Translate', 'Bing Translate', 'Ernie'], label=i18n.get('translation_method'), value='Google Translate')
                target_language = gr.Dropdown(['简体中文', '繁体中文', 'English', 'Cantonese', 'Japanese', 'Korean'], label=i18n.get('target_language'), value='Japanese')

                tts_method = gr.Dropdown(['xtts', 'cosyvoice', 'EdgeTTS'], label=i18n.get('tts_method'), value='EdgeTTS')
                tts_target_language = gr.Dropdown(['中文', 'English', '粤语', 'Japanese', 'Korean', 'Spanish', 'French'], label=i18n.get('tts_target_language'), value='Japanese')
                edgetts_voice = gr.Dropdown(SUPPORT_VOICE, value='ja-JP-NanamiNeural', label=i18n.get('edgetts_voice'), visible=True)

                add_subtitles = gr.Checkbox(label=i18n.get('add_subtitles'), value=True)
                speed_factor = gr.Slider(minimum=0.5, maximum=2, step=0.05, label=i18n.get('speed_factor'), value=1.00)
                fps = gr.Slider(minimum=1, maximum=60, step=1, label=i18n.get('fps'), value=30)
                bg_music = gr.Audio(label=i18n.get('bg_music'), sources=['upload'])
                bg_music_volume = gr.Slider(minimum=0, maximum=1, step=0.05, label=i18n.get('bg_music_volume'), value=0.5)
                video_volume = gr.Slider(minimum=0, maximum=1, step=0.05, label=i18n.get('video_volume'), value=1.0)
                resolution_out = gr.Radio(['4320p', '2160p', '1440p', '1080p', '720p', '480p', '360p', '240p', '144p'], label=i18n.get('resolution'), value='1080p')

                max_workers = gr.Slider(minimum=1, maximum=100, step=1, label=i18n.get('max_workers'), value=1)
                max_retries = gr.Slider(minimum=1, maximum=10, step=1, label=i18n.get('max_retries'), value=3)
                
                run_button = gr.Button(i18n.get('one_click_process'), variant="primary")

            with gr.Column():
                synthesis_status = gr.Text(label=i18n.get('synthesis_status'))
                synthesis_result = gr.Video(label=i18n.get('synthesis_result'))

        def update_tts_visibility(method):
            if method == 'EdgeTTS':
                return gr.update(visible=True)
            else:
                return gr.update(visible=False)

        tts_method.change(fn=update_tts_visibility, inputs=tts_method, outputs=edgetts_voice)

        run_button.click(
            fn=do_everything,
            inputs=[
                video_output_folder, video_url, video_file, download_video_count, resolution,
                model, compute_device, number_of_shifts,
                asr_model_selection, whisperx_model_size, batch_size, separate_speakers, min_speakers, max_speakers,
                translation_method, target_language,
                tts_method, tts_target_language, edgetts_voice,
                add_subtitles, speed_factor, fps, bg_music, bg_music_volume, video_volume, resolution_out,
                max_workers, max_retries
            ],
            outputs=[synthesis_status, synthesis_result]
        )
    return interface

full_auto_interface = create_full_auto_interface()

# 下载视频接口
download_interface = gr.Interface(
    fn=download_from_url,
    inputs=[
        gr.Textbox(label=i18n.get('video_url'), placeholder=i18n.get('video_url_placeholder'), 
                   value='https://www.bilibili.com/video/BV1kr421M7vz/'),
        gr.Textbox(label=i18n.get('video_output_folder'), value='videos'),
        gr.Radio(['4320p', '2160p', '1440p', '1080p', '720p', '480p', '360p', '240p', '144p'], label=i18n.get('resolution'), value='1080p'),
        gr.Slider(minimum=1, maximum=100, step=1, label=i18n.get('download_video_count'), value=5),
        # gr.Checkbox(label='单个视频', value=False),
    ],
    outputs=[
        gr.Textbox(label=i18n.get('download_status')), 
        gr.Video(label=i18n.get('example_video')), 
        gr.Json(label=i18n.get('download_info'))
    ],
    allow_flagging='never',
)

# 人声分离接口
demucs_interface = gr.Interface(
    fn=separate_all_audio_under_folder,
    inputs=[
        gr.Textbox(label=i18n.get('video_folder'), value='videos'),
        gr.Radio(['htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'hdemucs_mmi', 'mdx', 'mdx_extra', 'mdx_q', 'mdx_extra_q', 'SIG'], label=i18n.get('model'), value='htdemucs_ft'),
        gr.Radio(['auto', 'cuda', 'cpu'], label=i18n.get('compute_device'), value='auto'),
        gr.Checkbox(label=i18n.get('show_progress'), value=True),
        gr.Slider(minimum=0, maximum=10, step=1, label=i18n.get('number_of_shifts'), value=5),
    ],
    outputs=[
        gr.Text(label=i18n.get('separation_status')), 
        gr.Audio(label=i18n.get('vocal_audio')), 
        gr.Audio(label=i18n.get('accompaniment_audio'))
    ],
    allow_flagging='never',
)

# AI智能语音识别接口
asr_inference = gr.Interface(
    fn=transcribe_all_audio_under_folder,
    inputs=[
        gr.Textbox(label=i18n.get('video_folder'), value='videos'),
        gr.Dropdown(['WhisperX', 'FunASR'], label=i18n.get('asr_model_selection'), value='WhisperX'),
        gr.Radio(['large', 'medium', 'small', 'base', 'tiny'], label=i18n.get('whisperx_model_size'), value='large'),
        gr.Radio(['auto', 'cuda', 'cpu'], label=i18n.get('compute_device'), value='auto'),
        gr.Slider(minimum=1, maximum=128, step=1, label=i18n.get('batch_size'), value=32),
        gr.Checkbox(label=i18n.get('separate_speakers'), value=True),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label=i18n.get('min_speakers'), value=None),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label=i18n.get('max_speakers'), value=None),
    ],
    outputs=[
        gr.Text(label=i18n.get('asr_status')), 
        gr.Json(label=i18n.get('asr_details'))
    ],
    allow_flagging='never',
)

# 翻译字幕接口
translation_interface = gr.Interface(
    fn=translate_all_transcript_under_folder,
    inputs=[
        gr.Textbox(label=i18n.get('video_folder'), value='videos'),
        gr.Dropdown(['OpenAI', 'LLM', 'Google Translate', 'Bing Translate', 'Ernie'], label=i18n.get('translation_method'), value='LLM'),
        gr.Dropdown(['简体中文', '繁体中文', 'English', 'Cantonese', 'Japanese', 'Korean'], label=i18n.get('target_language'), value='Japanese'),
    ],
    outputs=[
        gr.Text(label=i18n.get('translation_status')), 
        gr.Json(label=i18n.get('summary_result')), 
        gr.Json(label=i18n.get('translation_result'))
    ],
    allow_flagging='never',
)

def create_tts_interface():
    with gr.Blocks() as interface:
        with gr.Row():
            with gr.Column():
                video_folder = gr.Textbox(label=i18n.get('video_folder'), value='videos')
                tts_method = gr.Dropdown(['xtts', 'cosyvoice', 'EdgeTTS'], label=i18n.get('tts_method'), value='xtts')
                tts_target_language = gr.Dropdown(['中文', 'English', '粤语', 'Japanese', 'Korean', 'Spanish', 'French'], label=i18n.get('tts_target_language'), value='Japanese')
                edgetts_voice = gr.Dropdown(SUPPORT_VOICE, value='ja-JP-NanamiNeural', label=i18n.get('edgetts_voice'), visible=False)
                
                run_button = gr.Button(i18n.get('one_click_process'), variant="primary")

            with gr.Column():
                synthesis_status = gr.Text(label=i18n.get('synthesis_status'))
                tts_results = gr.Audio(label=i18n.get('tts_results'))
                vocal_audio = gr.Audio(label=i18n.get('vocal_audio'))

        def update_tts_visibility(method):
            if method == 'EdgeTTS':
                return gr.update(visible=True)
            else:
                return gr.update(visible=False)

        tts_method.change(fn=update_tts_visibility, inputs=tts_method, outputs=edgetts_voice)

        run_button.click(
            fn=generate_all_wavs_under_folder,
            inputs=[video_folder, tts_method, tts_target_language, edgetts_voice],
            outputs=[synthesis_status, tts_results, vocal_audio]
        )
    return interface

tts_interface = create_tts_interface()

# 视频合成接口
synthesize_video_interface = gr.Interface(
    fn=synthesize_all_video_under_folder,
    inputs=[
        gr.Textbox(label=i18n.get('video_folder'), value='videos'),
        gr.Checkbox(label=i18n.get('add_subtitles'), value=True),
        gr.Slider(minimum=0.5, maximum=2, step=0.05, label=i18n.get('speed_factor'), value=1.00),
        gr.Slider(minimum=1, maximum=60, step=1, label=i18n.get('fps'), value=30),
        gr.Audio(label=i18n.get('bg_music'), sources=['upload'], type='filepath'),
        gr.Slider(minimum=0, maximum=1, step=0.05, label=i18n.get('bg_music_volume'), value=0.5),
        gr.Slider(minimum=0, maximum=1, step=0.05, label=i18n.get('video_volume'), value=1.0),
        gr.Radio(['4320p', '2160p', '1440p', '1080p', '720p', '480p', '360p', '240p', '144p'], label=i18n.get('resolution'), value='1080p'),

    ],
    outputs=[
        gr.Text(label=i18n.get('synthesis_status')), 
        gr.Video(label=i18n.get('synthesized_video'))
    ],
    allow_flagging='never',
)

linly_talker_interface = gr.Interface(
    fn=lambda: None,
    inputs=[
        gr.Textbox(label=i18n.get('video_folder'), value='videos'),
        gr.Dropdown(['Wav2Lip', 'Wav2Lipv2','SadTalker'], label=i18n.get('dubbing_method'), value='Wav2Lip'),
    ],      
    outputs=[
        gr.Markdown(value=i18n.get('under_construction') + " [https://github.com/Kedreamix/Linly-Talker](https://github.com/Kedreamix/Linly-Talker)"),
        gr.Text(label=i18n.get('synthesis_status')),
        gr.Video(label=i18n.get('synthesized_video'))
    ],
)

my_theme = gr.themes.Soft()
# 应用程序界面
app = gr.TabbedInterface(
    theme=my_theme,
    interface_list=[
        full_auto_interface,
        download_interface,
        demucs_interface,
        asr_inference,
        translation_interface,
        tts_interface,
        synthesize_video_interface,
        linly_talker_interface
    ],
    tab_names=[
        i18n.get('tab_full_auto'), 
        i18n.get('tab_download'), i18n.get('tab_demucs'), i18n.get('tab_asr'), i18n.get('tab_translation'), i18n.get('tab_tts'), i18n.get('tab_synthesize'),
        i18n.get('tab_talker')],
    title=i18n.get('app_title')
)

if __name__ == '__main__':
    app.launch(
        server_name="127.0.0.1", 
        server_port=6006,
        share=True,
        inbrowser=True
    )