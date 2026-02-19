import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.append(os.getcwd())

from src.modules.translation.providers.groq_api import GroqTranslator

def test_groq():
    t = GroqTranslator()
    text = "选到梅毒的那杯就可以成为我的弟子"
    summary = "Context about a master teaching a disciple."
    target_language = "vi"
    
    s_prompt = f"Translate the following text to {target_language}. Context: {summary}. Return JSON {{ \"translation\": \"text\" }}."
    msgs = [{"role": "system", "content": s_prompt}, {"role": "user", "content": text}]
    
    print(f"Sending prompt: {s_prompt}")
    print(f"Input text: {text}")
    
    try:
        res = t.translate(msgs)
        print(f"Raw Response: {res}")
        
        if res:
            content = res
            if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: content = content.split("```")[1].split("```")[0].strip()
            print(f"Cleaned content: {content}")
            
            try:
                res_json = json.loads(content)
                final = res_json.get('translation', res_json.get('text', res))
                print(f"Parsed Translation: {final}")
            except Exception as e:
                print(f"JSON Parse Error: {e}")
    except Exception as e:
        print(f"API Error: {e}")

if __name__ == "__main__":
    test_groq()
