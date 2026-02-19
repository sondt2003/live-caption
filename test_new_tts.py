import sys
import os

# Thêm đường dẫn src vào hệ thống
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from modules.tts.factory import TTSFactory

def test_edge_tts():
    print("Testing Edge-TTS...")
    engine = TTSFactory.get_tts_engine('edge')
    output_path = "test_edge.wav"
    if os.path.exists(output_path): os.remove(output_path)
    engine.generate("Hello, this is a test of Edge TTS.", output_path, target_language='en')
    if os.path.exists(output_path):
        print(f"Edge-TTS success: {output_path}")
    else:
        print("Edge-TTS failed.")

def test_gtts():
    print("Testing gTTS...")
    engine = TTSFactory.get_tts_engine('gtts')
    output_path = "test_gtts.wav"
    if os.path.exists(output_path): os.remove(output_path)
    engine.generate("Hello, this is a test of Google TTS.", output_path, target_language='en')
    if os.path.exists(output_path):
        print(f"gTTS success: {output_path}")
    else:
        print("gTTS failed.")

# Azure and OpenAI require API keys, so they might fail without environment variables
def test_openai_tts():
    if not os.environ.get('OPENAI_API_KEY'):
        print("Skipping OpenAI TTS test (No API key)")
        return
    print("Testing OpenAI TTS...")
    engine = TTSFactory.get_tts_engine('openai')
    output_path = "test_openai.wav"
    if os.path.exists(output_path): os.remove(output_path)
    engine.generate("Hello, this is a test of OpenAI TTS.", output_path, target_language='en')
    if os.path.exists(output_path):
        print(f"OpenAI TTS success: {output_path}")
    else:
        print("OpenAI TTS failed.")

def test_azure_tts():
    if not os.environ.get('AZURE_SPEECH_KEY'):
        print("Skipping Azure TTS test (No API key)")
        return
    print("Testing Azure TTS...")
    engine = TTSFactory.get_tts_engine('azure')
    output_path = "test_azure.wav"
    if os.path.exists(output_path): os.remove(output_path)
    engine.generate("Hello, this is a test of Azure TTS.", output_path, target_language='en')
    if os.path.exists(output_path):
        print(f"Azure TTS success: {output_path}")
    else:
        print("Azure TTS failed.")

if __name__ == "__main__":
    test_edge_tts()
    test_gtts()
    test_openai_tts()
    test_azure_tts()
