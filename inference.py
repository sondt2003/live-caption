import os
import warnings

# Suppress TensorFlow warnings BEFORE importing TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

# Suppress Python warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress SpeechBrain logging
import logging
logging.getLogger('speechbrain').setLevel(logging.ERROR)

import time as tm
from concurrent.futures import ThreadPoolExecutor

print("Start Processing...")

from pyannote.audio import Pipeline
from audio_separator.separator import Separator
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS
from pydub import AudioSegment
import shutil
import subprocess
import torch
from speechbrain.inference.interfaces import foreign_class
from deepface import DeepFace
import numpy as np
import cv2
import json
import re
from groq import Groq
import sys, argparse
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
import warnings
from tools.utils import get_speaker
from tools.utils import extract_frames
from tools.utils import detect_and_crop_faces
from tools.utils import extract_and_save_most_common_face
from tools.utils import get_overlap
from faster_whisper import WhisperModel

        
nltk.download('punkt', quiet=True)  # Suppress download messages
load_dotenv()

parser = argparse.ArgumentParser(description='Choose between YouTube or video URL')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--yt_url', type=str, help='YouTube single video URL', default='')
group.add_argument('--video_url', type=str, help='Single video URL')

parser.add_argument('--source_language', type=str, help='Video source language', required=True)
parser.add_argument('--target_language', type=str, help='Video target language', required=True)
parser.add_argument('--whisper_model', type=str, help='Chose the whisper model based on your device requirements', default="medium")
parser.add_argument('--LipSync', type=bool, help='Lip synchronization of the resut audio to the synthesized video', default=False)
parser.add_argument('--Bg_sound', type=bool, help='Keep the background sound of the original video, though it might be slightly noisy', default=False)



args = parser.parse_args()



class VideoDubbing:
    def __init__(self, Video_path, source_language, target_language, 
                 LipSync=True, Voice_denoising = True, whisper_model="medium",
                 Context_translation = "API code here", huggingface_auth_token="API code here"):
        
        self.Video_path = Video_path
        self.source_language = source_language
        self.target_language = target_language
        self.LipSync = LipSync
        self.Voice_denoising = Voice_denoising
        self.whisper_model = whisper_model
        self.Context_translation = Context_translation
        self.huggingface_auth_token = huggingface_auth_token
        
        if not self.huggingface_auth_token or self.huggingface_auth_token == "API code here":
             print("\n[ERROR] Hugging Face token is missing! Please provide a valid token in the .env file (HF_TOKEN).")
             print("You can get a token from https://huggingface.co/settings/tokens")
             print("Ensure you have accepted the user agreement for 'pyannote/speaker-diarization' on Hugging Face.\n")
             sys.exit(1)

        def log_profile(step, duration):
            with open("workflow_profiling.log", "a") as f:
                f.write(f"[{tm.strftime('%Y-%m-%d %H:%M:%S')}] {step}: {duration:.2f} seconds\n")

        overall_start = tm.time()
        
        # Clean workspace directory completely on each run
        os.system("rm -rf workspace")
        os.system("mkdir -p workspace/audio workspace/results")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the pre-trained speaker diarization pipeline
        speakers_rolls = {}
        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                         use_auth_token=self.huggingface_auth_token).to(device)
            
            # Extract audio from video to WAV format (pyannote cannot read MP4 directly)
            print("Extracting audio for diarization...")
            audio = AudioSegment.from_file(self.Video_path, format="mp4")
            audio_file = "workspace/audio/diarization_input.wav"
            audio.export(audio_file, format="wav")
            
            # Apply the diarization pipeline on the audio file
            diarization_start = tm.time()
            diarization = pipeline(audio_file)
            log_profile("Speaker Diarization", tm.time() - diarization_start)
            
            # Print the diarization results
            for speech_turn, _, speaker in diarization.itertracks(yield_label=True):
                if abs(speech_turn.end - speech_turn.start) > 1.5:
                    print(f"Speaker {speaker}: from {speech_turn.start}s to {speech_turn.end}s")
                    speakers_rolls[(speech_turn.start, speech_turn.end)] = speaker
        except Exception as e:
            print(f"Diarization failed or HF_TOKEN missing: {e}")
            print("Proceeding with single speaker assumption.")
            audio = AudioSegment.from_file(self.Video_path, format="mp4")
            duration = len(audio) / 1000.0
            speakers_rolls[(0, duration)] = "SPEAKER_00"
            audio_file = self.Video_path  # Use video directly
        


        if self.LipSync:
            face_detection_start = tm.time()  # Start timing face detection
            # Load the video file
            video = cv2.VideoCapture(self.Video_path)
            
            # Get frames per second (FPS)
            fps = video.get(cv2.CAP_PROP_FPS)
            
            # Get total number of frames
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            video.release()
            
            
            
            
            frame_per_speaker = []
            
            for i in range(total_frames):
                time = i/round(fps)
                frame_speaker = get_speaker(time, speakers_rolls)
                frame_per_speaker.append(frame_speaker)
                # print(time)
            
            os.system("rm -r workspace/speakers_image")
            os.system("mkdir workspace/speakers_image")
            
            
            
            # Specify the video path and output folder
            output_folder = "workspace/speakers_image"
            # Call the function
            extract_frames(self.Video_path, output_folder, speakers_rolls)
            
            # Initialize the MTCNN face detector
            haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            
            # Load the pre-trained Haar Cascade model for face detection
            face_cascade = cv2.CascadeClassifier(haar_cascade_path)
            
            # Function to detect and crop faces
            
            # Path to the folder containing speaker images
            speaker_images_folder = "workspace/speakers_image"
            
            # Iterate through speaker subfolders
            for speaker_folder in os.listdir(speaker_images_folder):
                speaker_folder_path = os.path.join(speaker_images_folder, speaker_folder)
            
                if os.path.isdir(speaker_folder_path):
                    # Process each image in the speaker folder
                    for image_name in os.listdir(speaker_folder_path):
                        image_path = os.path.join(speaker_folder_path, image_name)
            
                        # Detect and crop faces from the image
                        if not detect_and_crop_faces(image_path, face_cascade):
                            # If no face is detected, delete the image
                            os.remove(image_path)
                            print(f"Deleted {image_path} due to no face detected.")
                        else:
                            print(f"Face detected and cropped: {image_path}")
            
            
        
            
            speaker_images_folder = "workspace/speakers_image"
            for speaker_folder in os.listdir(speaker_images_folder):
                speaker_folder_path = os.path.join(speaker_images_folder, speaker_folder)
            
                print(f"Processing images in folder: {speaker_folder}")
                # Try to extract most common face, fallback to first face if DeepFace fails
                max_image_path = extract_and_save_most_common_face(speaker_folder_path)
                
                # If DeepFace face grouping failed, use the first detected face
                if max_image_path is None:
                    print(f"Warning: DeepFace face grouping failed for {speaker_folder}. Using first detected face.")
                    # Find first image in the folder
                    images = [f for f in os.listdir(speaker_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if images:
                        first_image = os.path.join(speaker_folder_path, images[0])
                        max_image_path = os.path.join(speaker_folder_path, "max_image.jpg")
                        shutil.copy(first_image, max_image_path)
                        print(f"Using first detected face: {first_image}")
                    else:
                        print(f"Error: No face images found for {speaker_folder}")
                        continue

            for root, dirs, files in os.walk(speaker_images_folder):
                for file in files:
                    # Check if the file is not 'max_image.jpg'
                    if file != "max_image.jpg":
                        # Construct full file path
                        file_path = os.path.join(root, file)
                        # Delete the file
                        os.remove(file_path)
            
            
            
            # Save to a file
            with open('workspace/frame_per_speaker.json', 'w') as f:
                json.dump(frame_per_speaker, f)
            
            
            if os.path.exists("Wav2Lip/frame_per_speaker.json"):
                os.remove("Wav2Lip/frame_per_speaker.json")
            shutil.copyfile('workspace/frame_per_speaker.json', "Wav2Lip/frame_per_speaker.json")
            
            
            if os.path.exists("Wav2Lip/speakers_image"):
                shutil.rmtree("Wav2Lip/speakers_image")
            shutil.copytree("workspace/speakers_image", "Wav2Lip/speakers_image")
            
            log_profile("Face Detection & Extraction", tm.time() - face_detection_start)

            
        ###############################################################################
        
        os.system("rm -r workspace/speakers_audio")
        os.system("mkdir workspace/speakers_audio")
        
        speakers = set(list(speakers_rolls.values()))
        audio = AudioSegment.from_file(audio_file, format="mp4")
        
        for speaker in speakers:
            speaker_audio = AudioSegment.empty()
            for key, value in speakers_rolls.items():
                if speaker == value:
                    start = int(key[0])*1000
                    end = int(key[1])*1000
                    
                    speaker_audio += audio[start:end]
                    
        
            speaker_audio.export(f"workspace/speakers_audio/{speaker}.wav", format="wav")
        
        most_occured_speaker= max(list(speakers_rolls.values()),key=list(speakers_rolls.values()).count)
        
        transcription_start = tm.time()
        model = WhisperModel(self.whisper_model, device='cpu', compute_type="int8")
        print(f"Starting transcription with model: {self.whisper_model} on CPU...")
        segments, info = model.transcribe(self.Video_path, word_timestamps=True)
        
        time_stamped = []
        full_text = []
        
        # Process segments one by one for progress visibility
        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            for word in segment.words:
                time_stamped.append([word.word, word.start, word.end])
                full_text.append(word.word)
        log_profile("Transcription (Whisper)", tm.time() - transcription_start)
        
        full_text_str = "".join(full_text)
        full_text = "".join(full_text)       
        # Decompose Long Sentences

        
        
        # Tokenize the text into sentences
        tokenized_sentences = sent_tokenize(full_text)
        sentences = []
        
        # Print the sentences
        for i, sentence in enumerate(tokenized_sentences):
            sentences.append(sentence)

        
        time_stamped_sentances = {}
        count_sentances = {}
        print(sentences)
        letter = 0
        for i in range(len(sentences)):
            tmp = []
            starts = []
            
            for j in range(len(sentences[i])):
                letter += 1
                tmp.append(sentences[i][j])
                
                f = 0
                for k in range(len(time_stamped)):
                    for m in range(len(time_stamped[k][0])):
                        f += 1
                        
                        if f == letter:
        
                            starts.append(time_stamped[k][1])
                       
                            starts.append(time_stamped[k][2])
            letter += 1               
                            
            time_stamped_sentances["".join(tmp)] = [min(starts), max(starts)]
            count_sentances[i+1] = "".join(tmp)

        record = []
        for sentence in time_stamped_sentances:
            record.append([sentence, time_stamped_sentances[sentence][0], time_stamped_sentances[sentence][1]])
        

        
        new_record = record
        
        # Audio Emotions Analysis
        
        classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier", run_opts={"device":f"{device}"})
        
        emotion_dict = {'neu': 'Neutral',
                        'ang' : 'Angry',
                        'hap' : 'Happy',
                        'sad' : 'Sad',
                        'None': None}
    

        
        self.cached_translator = None
        self.cached_tokenizer = None

        def get_translator():
            if not self.cached_translator:
                target_language = self.target_language
                if self.source_language == 'tr':
                    model_name = f"Helsinki-NLP/opus-mt-trk-{target_language}"
                elif target_language == 'tr':
                    model_name = f"Helsinki-NLP/opus-mt-{self.source_language}-trk"
                elif self.source_language == 'zh-cn':
                    model_name = f"Helsinki-NLP/opus-mt-zh-{target_language}"
                elif target_language == 'zh-cn':
                    model_name = f"Helsinki-NLP/opus-mt-{self.source_language}-zh"
                else:
                    model_name = f"Helsinki-NLP/opus-mt-{self.source_language}-{target_language}"
                
                print(f"Loading Translation Model: {model_name}...")
                self.cached_tokenizer = MarianTokenizer.from_pretrained(model_name)
                self.cached_translator = MarianMTModel.from_pretrained(model_name).to('cpu')
            return self.cached_tokenizer, self.cached_translator

        def translate(sentence, target_language):
            if not self.Context_translation:
                tokenizer, model = get_translator()
                inputs = tokenizer([sentence], return_tensors="pt", padding=True).to('cpu')
                translated = model.generate(**inputs)
                return tokenizer.decode(translated[0], skip_special_tokens=True)
            else:
                client = Groq(api_key=self.Context_translation)
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": f"Professional translator. Translate to {target_language}: {sentence}. Format: [[sentence translation: <translation>]]"}],
                    model="llama-3.3-70b-versatile",  # Updated from deprecated llama3-70b-8192
                )
                match = re.search(r'\[\[sentence translation: (.*?)\]\]', chat_completion.choices[0].message.content)
                return match.group(1) if match else "Error in translation"

        def batch_translate(sentences, target_language, batch_size=10):
            if not self.Context_translation:
                tokenizer, model = get_translator()
                print(f"Translating {len(sentences)} sentences on CPU in batches of {batch_size}...")
                
                translated_sentences = []
                total_batches = (len(sentences) + batch_size - 1) // batch_size
                
                for i in range(0, len(sentences), batch_size):
                    batch = sentences[i : i + batch_size]
                    current_batch = (i // batch_size) + 1
                    print(f"Translating batch {current_batch}/{total_batches} ({len(batch)} sentences)...")
                    
                    try:
                        inputs = tokenizer(batch, return_tensors="pt", padding=True).to('cpu')
                        with torch.no_grad():
                            translated = model.generate(**inputs)
                        batch_results = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
                        translated_sentences.extend(batch_results)
                    except Exception as e:
                        print(f"Error translating batch {current_batch}: {e}")
                        # Fallback: add original text or empty strings to keep alignment? 
                        # For now let's just re-raise or handle gracefully. 
                        # Re-raising might be better to avoid misalignment.
                        raise e
                        
                return translated_sentences
            else:
                return [translate(s, target_language) for s in sentences]
                    
               

        
        records = []
        
        all_original_texts = [r[0] for r in new_record]
        print(f"Translating {len(all_original_texts)} segments in batch...")
        all_translated_texts = batch_translate(all_original_texts, self.target_language)

        loop_start = tm.time()
        for i in range(len(new_record)):
            final_sentance = all_original_texts[i]
            translated = all_translated_texts[i]
            speaker = most_occured_speaker
            
            max_overlap = 0
        
            # Check overlap with each speaker's time range
            for key, value in speakers_rolls.items():
                speaker_start =  int(key[0])
                speaker_end = int(key[1])
                
                # Calculate overlap
                overlap = get_overlap((new_record[i][1], new_record[i][2]), (speaker_start, speaker_end))
                
                # Update speaker if this overlap is greater than previous ones
                if overlap > max_overlap:
                    max_overlap = overlap
                    speaker = value
                    
            start = int(new_record[i][1]) *1000
            end = int(new_record[i][2]) *1000
        
            try:
                audio[start:end].export("workspace/audio/emotions.wav", format="wav")      
                out_prob, score, index, text_lab = classifier.classify_file("workspace/audio/emotions.wav")
                os.remove("workspace/audio/emotions.wav")
            except:
                text_lab = ['None']
            
            records.append([translated, final_sentance, new_record[i][1], new_record[i][2], speaker, emotion_dict[text_lab[0]]])
            print(translated, final_sentance, new_record[i][1], new_record[i][2], speaker, emotion_dict[text_lab[0]])
        log_profile("Translation & Emotion Analysis (Full Loop)", tm.time() - loop_start)
        
        
        
        os.environ["COQUI_TOS_AGREED"] = "1"
        if device == "cuda":
                tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        else:
                tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        #!tts --model_name "tts_models/multilingual/multi-dataset/xtts_v2"  --list_speaker_idxs
        
        os.system("rm -r workspace/audio_chunks")
        os.system("rm -r workspace/su_audio_chunks")
        os.system("mkdir workspace/audio_chunks")
        os.system("mkdir workspace/su_audio_chunks")

        natural_scilence = records[0][2]
        previous_silence_time = 0
        
        if natural_scilence >= 0.8:
            previous_silence_time = 0.8
            natural_scilence -= 0.8
        else:
            previous_silence_time = natural_scilence
            natural_scilence = 0   
            
        combined = AudioSegment.silent(duration=natural_scilence*1000) 

        tip = 350

        def truncate_text(text, max_tokens=50):
                words = text.split()
                if len(words) <= max_tokens:
                        return text
                return ' '.join(words[:max_tokens]) + '...'
        
        tts_start = tm.time()
        print(f"Starting parallel TTS generation for {len(records)} segments...")

        def generate_tts_chunk(i):
            record = records[i]
            wav_path = f"workspace/audio_chunks/{i}.wav"
            tts.tts_to_file(text=truncate_text(record[0]),
                            file_path=wav_path,
                            speaker_wav=f"workspace/speakers_audio/{record[4]}.wav",
                            language=self.target_language,
                            emotion=record[5],
                            speed=2)
            return i

        # Limit workers to avoid CPU choking on heavy XTTS
        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
            list(executor.map(generate_tts_chunk, range(len(records))))

        for i in range(len(records)):
            print(f"Processing chunk {i}/{len(records)} alignment...")
            audio = AudioSegment.from_file(f"workspace/audio_chunks/{i}.wav")
            audio = audio[:len(audio)-tip]
            audio.export(f"workspace/audio_chunks/{i}.wav", format="wav")
            
            
            lt = len(audio) / 1000.0 
            lo =  max(records[i][3] - records[i][2], 0)
            theta = lo/lt
          
            input_file = f"workspace/audio_chunks/{i}.wav"
            output_file = f"workspace/su_audio_chunks/{i}.wav"

           
            if theta <1 and theta > 0.44:
                print('############################')
                theta_prim = (lo+previous_silence_time)/lt
                command = f"ffmpeg -i {input_file} -filter:a 'atempo={1/theta_prim}' -vn {output_file}"
                process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if process.returncode != 0:
                    sc = lo  + previous_silence_time
                    silence = AudioSegment.silent(duration=(sc*1000))
                    silence.export(output_file, format="wav")
            elif theta < 0.44:
                silence = AudioSegment.silent(duration=((lo+previous_silence_time)*1000))
                silence.export(output_file, format="wav")
            else:
                silence = AudioSegment.silent(duration=(previous_silence_time*1000))
                audio = silence  + audio
                audio.export(output_file, format="wav")
        
                
            audio = AudioSegment.from_file(output_file)
            lt = len(audio) / 1000.0
            lo =  records[i][3]-records[i][2]+ previous_silence_time
            if i+1 < len(records):
                natural_scilence = max(records[i+1][2]-records[i][3], 0) 
                if natural_scilence >= 0.8:
                    previous_silence_time = 0.8
                    natural_scilence -= 0.8
                else:
                    previous_silence_time = natural_scilence
                    natural_scilence = 0
                
                    
                silence = AudioSegment.silent(duration=((max(lo-lt,0)+natural_scilence)*1000))
                audio_with_silence = audio + silence
                audio_with_silence.export(output_file, format="wav")
            else:
                silence = AudioSegment.silent(duration=(max(lo-lt,0)*1000))
                audio_with_silence = audio + silence
                audio_with_silence.export(output_file, format="wav")
            
            audio_with_silence.export(output_file, format="wav")
            
            print(f"Chunk {i} alignment: diff={lo-lt:.2f}s")
            del audio
        log_profile("TTS Generation & Audio Alignment (Full Loop)", tm.time() - tts_start)
        
       
        
        # Get all the audio files from the folder
        audio_files = [f for f in os.listdir("su_audio_chunks") if f.endswith(('.mp3', '.wav', '.ogg'))]
        
        # Sort files to concatenate them in order, if necessary
        audio_files.sort(key=lambda x: int(x.split('.')[0]))  # Modify sorting logic if needed (e.g., based on filenames)
        
        # Loop through and concatenate each audio file
        for audio_file in audio_files:
            file_path = os.path.join("su_audio_chunks", audio_file)
            audio_segment = AudioSegment.from_file(file_path)
            combined += audio_segment  # Append audio to the combined segment
        
        
        audio = AudioSegment.from_file(self.Video_path)
        total_length = len(audio) / 1000.0 
        silence = AudioSegment.silent(duration=abs(total_length - records[-1][3])*1000)
        combined += silence
        # Export the combined audio to the output file
        combined.export("workspace/audio/output.wav", format="wav")

        
        # Initialize Spleeter with the 2stems model (vocals + accompaniment)
        audio_separation_start = tm.time()  # Start timing audio separation
        separator = Separator()

        # Load a model
        separator.load_model(model_filename='2_HP-UVR.pth')
        output_file_paths = separator.separate(self.Video_path)[0]
        log_profile("Audio Separation (Vocals/Background)", tm.time() - audio_separation_start)

      
        
        
        audio1 = AudioSegment.from_file("workspace/audio/output.wav")
        audio2 = AudioSegment.from_file(output_file_paths)
        combined_audio = audio1.overlay(audio2)
        
        # Export the combined audio file
        combined_audio.export("workspace/audio/combined_audio.wav", format="wav")
        
        
        # Video and Audio Overlay
        
        command = f"ffmpeg -i '{self.Video_path}' -i audio/combined_audio.wav -c:v copy -map 0:v:0 -map 1:a:0 -shortest output_video.mp4"
        subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        shutil.move(output_file_paths, "workspace/audio/")
        
        
        
        if self.Voice_denoising:
            command = f"ffmpeg -i '{self.Video_path}' -i audio/output.wav -c:v copy -map 0:v:0 -map 1:a:0 -shortest denoised_video.mp4"
            subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if self.LipSync and self.Voice_denoising:
            lipsync_start = tm.time()  # Start timing lip sync
            os.system("pip install librosa==0.9.1 > /dev/null 2>&1")
            os.system("cd Wav2Lip && python inference.py --checkpoint_path 'wav2lip_gan.pth' --face '../denoised_video.mp4' --audio '../audio/output.wav' --face_det_batch_size 1 --wav2lip_batch_size 1")
            
        if self.LipSync and not self.Voice_denoising:
            os.system("pip install librosa==0.9.1 > /dev/null 2>&1")
            os.system("cd Wav2Lip && python inference.py --checkpoint_path 'wav2lip_gan.pth' --face '../output_video.mp4' --audio '../audio/combined_audio.wav' --face_det_batch_size 1 --wav2lip_batch_size 1")

			 
        if  self.LipSync and self.Voice_denoising:
            source_path = 'Wav2Lip/results/result_voice.mp4'
            destination_folder = 'workspace/results'

            shutil.move(source_path, destination_folder)
            os.remove('output_video.mp4')
            shutil.move('denoised_video.mp4', destination_folder)

        elif self.LipSync and not self.Voice_denoising:
            source_path = 'Wav2Lip/results/result_voice.mp4'
            destination_folder = 'workspace/results'

            shutil.move(source_path, destination_folder)
            os.remove('output_video.mp4')
            os.remove('denoised_video.mp4')
            log_profile("Lip Sync (Wav2Lip)", tm.time() - lipsync_start)
		
        elif not self.LipSync and self.Voice_denoising:
            source_path = 'denoised_video.mp4'
            destination_folder = 'workspace/results'

            shutil.move(source_path, destination_folder)
            os.remove('output_video.mp4')
        else:
            source_path = 'output_video.mp4'
            destination_folder = 'workspace/results'

            shutil.move(source_path, destination_folder)
	
        log_profile("Total Execution Time", tm.time() - overall_start)
        
        # Print summary
        print("\n" + "="*60)
        print("✅ PROCESSING COMPLETE!")
        print("="*60)
        print(f"📊 Detailed timing: workflow_profiling.log")
        print(f"🎬 Output video: workspace/results/output.mp4")
        print("="*60 + "\n")
        # os.system('pip install -r requirements.txt > /dev/null 2>&1')	

def main():
	if os.path.exists("video_path.mp4"):
		os.system("rm video_path.mp4")
	video_path = None
	if args.yt_url:
		os.system(f"yt-dlp -f best -o 'video_path.mp4' --recode-video mp4 {args.yt_url}")
		video_path = "video_path.mp4"

	if not video_path:
		video_path = args.video_url
	
	vidubb = VideoDubbing(video_path, args.source_language, args.target_language, args.LipSync, not args.Bg_sound, args.whisper_model, os.getenv('GROQ_API_KEY'), os.getenv('HF_TOKEN'))
	
if __name__ == '__main__':
	main()
  
