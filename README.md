# ViDubb - AI Video Dubbing System

Hệ thống dubbing video tự động sử dụng AI để dịch và đồng bộ giọng nói, môi miệng cho video.

## 🎯 Tính năng chính

- **Speaker Diarization**: Phân biệt nhiều người nói trong video
- **Speech-to-Text**: Chuyển đổi giọng nói thành văn bản
- **Translation**: Dịch văn bản sang ngôn ngữ đích
- **Text-to-Speech**: Tạo giọng nói mới bằng ngôn ngữ đích
- **Lip Sync**: Đồng bộ môi miệng với giọng nói mới
- **Background Sound Separation**: Tách và giữ lại âm thanh nền

---

## 🤖 Models & Libraries được sử dụng

### 1. **Speaker Diarization** (Phân biệt người nói)
- **Model**: `pyannote/speaker-diarization-3.1`
- **Dependency**: `pyannote/segmentation-3.0`, `pyannote/embedding`
- **Chức năng**: Xác định ai đang nói và khi nào
- **Output**: Timestamps cho từng speaker (SPEAKER_00, SPEAKER_01, ...)

### 2. **Speech Recognition** (Nhận dạng giọng nói)
- **Model**: `faster-whisper` (base/small/medium/large)
- **Backend**: OpenAI Whisper
- **Chức năng**: Chuyển audio thành text với timestamps
- **Output**: Transcript với word-level timestamps

### 3. **Face Detection & Recognition** (Nhận diện khuôn mặt)
- **Model**: 
  - Haar Cascade (OpenCV) - Face detection
  - DeepFace ArcFace - Face recognition & grouping
- **Chức năng**: Detect và group faces cho mỗi speaker
- **Output**: Representative face cho mỗi speaker

### 4. **Translation** (Dịch thuật)
**Option 1: MarianMT** (Default - Offline)
- **Model**: `Helsinki-NLP/opus-mt-{source}-{target}`
- **Chức năng**: Dịch văn bản giữa các ngôn ngữ
- **Ưu điểm**: Miễn phí, offline

**Option 2: Groq API** (Optional - Online)
- **Model**: `llama3-70b-8192`
- **Chức năng**: Context-aware translation
- **Ưu điểm**: Chất lượng cao hơn, hiểu ngữ cảnh

### 5. **Text-to-Speech** (Tổng hợp giọng nói)
- **Model**: `XTTS v2` (Coqui TTS)
- **Model Path**: `tts_models/multilingual/multi-dataset/xtts_v2`
- **Chức năng**: Clone giọng nói từ audio sample
- **Features**: 
  - Voice cloning từ speaker audio
  - Emotion control
  - Speed adjustment

### 6. **Emotion Recognition** (Nhận dạng cảm xúc)
- **Model**: `speechbrain/emotion-recognition-wav2vec2-IEMOCAP`
- **Chức năng**: Phát hiện cảm xúc trong giọng nói
- **Output**: Neutral, Angry, Happy, Sad

### 7. **Audio Separation** (Tách âm thanh)
- **Library**: `audio-separator`
- **Model**: `2_HP-UVR.pth`
- **Chức năng**: Tách vocals và background music
- **Output**: Vocals track + Instrumental track

### 8. **Lip Sync** (Đồng bộ môi)
- **Model**: `Wav2Lip GAN`
- **Model Path**: `Wav2Lip/wav2lip_gan.pth`
- **Chức năng**: Sync môi miệng với audio mới
- **Input**: Video + Audio
- **Output**: Video với môi đồng bộ

---

## 🔄 Luồng xử lý (Pipeline)

### **Phase 1: Audio Extraction & Speaker Diarization**
```
Input Video
    ↓
Extract Audio (pydub)
    ↓
Speaker Diarization (pyannote) → Timestamps cho mỗi speaker
    ↓
Split audio theo speaker → workspace/speakers_audio/SPEAKER_XX.wav
```

### **Phase 2: Face Detection & Extraction** (Nếu LipSync = True)
```
Input Video
    ↓
Extract frames theo speaker timestamps (cv2)
    ↓
Detect faces (Haar Cascade)
    ↓
Group faces (DeepFace ArcFace) → Tìm face phổ biến nhất
    ↓
Save representative face → workspace/speakers_image/SPEAKER_XX/max_image.jpg
```

### **Phase 3: Speech-to-Text**
```
Audio
    ↓
Whisper Transcription → Segments với word timestamps
    ↓
Sentence Tokenization (NLTK) → Chia thành câu
    ↓
Map timestamps → Mỗi câu có start/end time
```

### **Phase 4: Translation**
```
Source Text Sentences
    ↓
Batch Translation (MarianMT hoặc Groq)
    ↓
Translated Sentences (Target Language)
```

### **Phase 5: Emotion Analysis**
```
Audio segments
    ↓
Emotion Recognition (SpeechBrain)
    ↓
Emotion labels (Neutral/Angry/Happy/Sad)
```

### **Phase 6: Text-to-Speech Generation**
```
Translated Text + Speaker Audio Sample + Emotion
    ↓
XTTS v2 Voice Cloning
    ↓
Generated Audio Chunks → workspace/audio_chunks/
    ↓
Speed Adjustment (ffmpeg atempo) → Match original duration
    ↓
Add Silence Padding → workspace/su_audio_chunks/
    ↓
Concatenate All Chunks → workspace/audio/output.wav
```

### **Phase 7: Background Sound Mixing**
```
Original Video
    ↓
Audio Separation (audio-separator) → Vocals + Instrumental
    ↓
Mix: New Vocals + Original Instrumental
    ↓
Combined Audio → workspace/audio/combined_audio.wav
```

### **Phase 8: Video Assembly**
```
Original Video + New Audio
    ↓
ffmpeg merge → output_video.mp4
```

### **Phase 9: Lip Sync** (Nếu LipSync = True)
```
output_video.mp4 + New Audio + Face Data
    ↓
Wav2Lip GAN Processing
    ↓
Final Video với môi đồng bộ → workspace/results/result_voice.mp4
```

---

## 📁 Cấu trúc thư mục

```
ViDubb/
├── inference.py              # Main script
├── app.py                    # Gradio web UI
├── tools/
│   └── utils.py             # Helper functions
├── Wav2Lip/                 # Lip sync model
│   ├── wav2lip_gan.pth      # Pre-trained weights
│   └── inference.py         # Wav2Lip inference
├── workspace/               # Runtime data (gitignored)
│   ├── audio/              # Audio files
│   ├── audio_chunks/       # TTS output chunks
│   ├── su_audio_chunks/    # Speed-adjusted chunks
│   ├── speakers_audio/     # Per-speaker audio
│   ├── speakers_image/     # Per-speaker faces
│   └── results/            # Final output videos
├── .env                     # API tokens
└── requirements.txt         # Dependencies
```

---

## 🔑 Environment Variables

```bash
# Required for speaker diarization
HF_TOKEN=hf_your_huggingface_token

# Optional for better translation
Groq_TOKEN=gsk_your_groq_api_key
```

### Hugging Face Setup
Cần accept user agreements cho các models:
1. https://huggingface.co/pyannote/speaker-diarization-3.1
2. https://huggingface.co/pyannote/segmentation-3.0
3. https://huggingface.co/pyannote/embedding

Sau đó login CLI:
```bash
huggingface-cli login --token YOUR_HF_TOKEN
```

---

## 🚀 Usage

### Command Line
```bash
python inference.py \
  --video_url /path/to/video.mp4 \
  --source_language en \
  --target_language vi \
  --LipSync True \
  --Bg_sound True \
  --whisper_model base
```

### Web UI
```bash
python app.py
```
Mở browser tại `http://localhost:7860`

---

## 🎛️ Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--video_url` | Path or YouTube URL | Input video |
| `--source_language` | en, es, fr, de, etc. | Source language code |
| `--target_language` | en, es, fr, de, vi, etc. | Target language code |
| `--LipSync` | True/False | Enable lip sync |
| `--Bg_sound` | True/False | Keep background sound |
| `--whisper_model` | tiny/base/small/medium/large | Whisper model size |

---

## ⚙️ Technical Details

### GPU Support
- TensorFlow: Auto-detect CUDA
- PyTorch: Auto-detect CUDA
- Whisper: Can run on CPU or CUDA

### Performance
- **Whisper base**: ~1x realtime on CPU
- **XTTS v2**: ~2-3s per sentence on GPU
- **Wav2Lip**: ~0.5x realtime on GPU

### Memory Requirements
- Minimum: 8GB RAM
- Recommended: 16GB RAM + 4GB VRAM

---

## 🐛 Known Issues

1. **DeepFace Keras 3.x Compatibility**: 
   - Error: `'KerasHistory' object has no attribute 'layer'`
   - Workaround: Fallback to first detected face

2. **HF_TOKEN Requirements**:
   - Must accept all pyannote model agreements
   - Must login via CLI

3. **Long Videos**:
   - May take significant time to process
   - Consider using smaller Whisper models

---

## 📝 License

This project uses multiple open-source models with different licenses. Please check individual model licenses before commercial use.
