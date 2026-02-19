# -*- coding: utf-8 -*-
import json
import os
import re
import time
import traceback
from loguru import logger
from .factory import TranslatorFactory

def get_necessary_info(info: dict):
    return {
        'title': info['title'],
        'uploader': info['uploader'],
        'description': info['description'],
        'upload_date': info['upload_date'],
        'tags': info['tags'],
    }

def ensure_transcript_length(transcript, max_length=4000):
    mid = len(transcript)//2
    before, after = transcript[:mid], transcript[mid:]
    length = max_length//2
    return before[:length] + after[-length:]

def split_text_into_sentences(para):
    para = re.sub('([。！？\?])([^，。！？\?”’》])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^，。！？\?”’》])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^，。！？\?”’》])', r"\1\n\2", para)
    para = re.sub('([。！？\?][”’])([^，。！？\?”’》])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")

def translation_postprocess(result):
    result = re.sub(r'\（[^)]*\）', '', result)
    result = result.replace('...', '，')
    result = re.sub(r'(?<=\d),(?=\d)', '', result)
    result = result.replace('²', '的平方').replace(
        '————', '：').replace('——', '：').replace('°', '度')
    result = result.replace("AI", '人工智能')
    return result

def valid_translation(text, translation):
    if (translation.startswith('```') and translation.endswith('```')):
        translation = translation[3:-3]
        return True, translation_postprocess(translation)
    
    if (translation.startswith('“') and translation.endswith('”')) or (translation.startswith('"') and translation.endswith('"')):
        translation = translation[1:-1]
        return True, translation_postprocess(translation)

    # Simplified common patterns
    for pattern in ['：“', '："', ':"', ': "']:
        if pattern in translation and ('”' in translation or '"' in translation):
            sep = '”' if '”' in translation else '"'
            translation = translation.split(pattern)[-1].split(sep)[0]
            return True, translation_postprocess(translation)

    if len(text) <= 10 and len(translation) > 15:
        return False, 'Only translate the following sentence and give me the result.'
    
    forbidden = ['翻译', '译文', '简体中文', 'translate', 'translation']
    translation = translation.strip()
    for word in forbidden:
        if word in translation:
            return False, f"Don't include `{word}` in the translation."
    
    return True, translation_postprocess(translation)

def split_sentences(translation, use_char_based_end=True):
    output_data = []
    for item in translation:
        start = item['start']
        text = item['text']
        speaker = item['speaker']
        translation_text = item['translation']

        if not translation_text or len(translation_text.strip()) == 0:
            output_data.append({
                "start": round(start, 3), "end": round(item['end'], 3),
                "text": text, "speaker": speaker, "translation": translation_text or "未翻译"
            })
            continue

        sentences = split_text_into_sentences(translation_text)
        duration_per_char = (item['end'] - item['start']) / max(1, len(translation_text)) if use_char_based_end else 0

        for j, sentence in enumerate(sentences):
            # Only provide the full original text as a 'prompt' for the context of the first segment of a split
            # OR better: provide nothing if we don't have a 1:1 mapping to avoid hallucination.
            # For now, we provide the text but only for the first segment to reduce clutter.
            segment_text = text if j == 0 else "" 
            
            sentence_end = start + duration_per_char * len(sentence) if use_char_based_end else item['end']
            output_data.append({
                "start": round(start, 3), "end": round(sentence_end, 3),
                "text": segment_text, "speaker": speaker, "translation": sentence
            })
            if use_char_based_end: start = sentence_end
    return output_data

def summarize(info, transcript, target_language='vi', method='LLM'):
    text_content = ' '.join(line['text'] for line in transcript)
    text_content = ensure_transcript_length(text_content, max_length=2000)
    info_message = f'Title: "{info["title"]}" Author: "{info["uploader"]}". '
    
    translator = TranslatorFactory.get_translator(method, target_language)
    
    if method in ['Google Translate', 'Bing Translate', 'google', 'bing']:
        full_description = f'{info_message}\n{text_content}\n'
        summary_text = translator.translate([{'role': 'user', 'content': full_description}])
        return {
            'title': translator.translate([{'role': 'user', 'content': info['title']}]),
            'author': info['uploader'], 'summary': summary_text, 'language': target_language
        }

    full_description = f'The following is the full content of the video:\n{info_message}\n{text_content}\nDetailedly Summarize the video in JSON format:\n```json\n{{"title": "", "summary": ""}}\n```'
    
    system_prompt = f'You are a expert in the field of this video. Please summary the video in JSON format.\n```json\n{{"title": "the title", "summary": "the summary"}}\n```'
    
    success = False
    summary_data = {}
    for retry in range(5):
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': full_description}
            ]
            response = translator.translate(messages)
            # Find the first JSON-like structure
            match = re.search(r'\{.*\}', response.replace('\n', ' '), re.DOTALL)
            if not match: 
                # Try harder to find JSON if there are backticks
                if "```json" in response:
                    json_str = response.split("```json")[-1].split("```")[0].strip()
                    summary_data = json.loads(json_str)
                    success = True
                    break
                raise Exception("No JSON found")
            summary_data = json.loads(match.group())
            success = True
            break
        except Exception as e:
            logger.warning(f'Summarize failed (retry {retry}): {e}')
            if retry == 4:
                logger.error(f"Full response from LLM that failed: {response if 'response' in locals() else 'No response'}")
            time.sleep(1)
            
    if not success: 
        logger.error("Summarization process failed after 5 retries. Using fallback empty summary.")
        return {
            'title': info.get('title', 'Unknown'),
            'author': info.get('uploader', 'Unknown'),
            'summary': '',
            'tags': info.get('tags', []),
            'language': target_language
        }

    # Translate summary components if not already in target language
    try:
        trans_messages = [
            {'role': 'system', 'content': f'Translate video metadata into {target_language} JSON format.'},
            {'role': 'user', 'content': f'Title: {summary_data.get("title")}\nSummary: {summary_data.get("summary")}\nTags: {info["tags"]}'}
        ]
        # Logic to call translator.translate(trans_messages) could go here if needed, 
        # but for now we follow the existing pattern of returning what we got or defaulting.
    except:
        pass

    return {
        'title': summary_data.get('title', info['title']),
        'author': info['uploader'],
        'summary': summary_data.get('summary', ''),
        'tags': info['tags'],
        'language': target_language
    }

def _translate(summary, transcript, target_language='vi', method='LLM'):
    info = f'Video: "{summary["title"]}". Summary: {summary["summary"]}.'
    translator = TranslatorFactory.get_translator(method, target_language)
    
    fixed_message = [
        {'role': 'system', 'content': f'You are a language expert. Task: Translate video transcript into {target_language}. Model after: "{summary["title"]}".'}
    ]
    
    history = []
    full_translation = []
    for line in transcript:
        text = line['text']
        translation = ""
        for retry in range(5):
            messages = fixed_message + history[-10:] + [{'role': 'user', 'content': f'Translate:"{text}"'}]
            try:
                response = translator.translate(messages)
                translation = response.replace('\n', '')
                success, translation = valid_translation(text, translation)
                if not success: raise Exception('Invalid translation')
                break
            except Exception as e:
                logger.error(f'Translation error: {e}')
                time.sleep(0.5)
        
        full_translation.append(translation)
        history.append({'role': 'user', 'content': f'Translate:"{text}"'})
        history.append({'role': 'assistant', 'content': translation})
        
    return full_translation

def translate(method, folder, target_language='vi'):
    if os.path.exists(os.path.join(folder, 'translation.json')):
        logger.info(f'Translation already exists in {folder}')
        return True
    
    info_path = os.path.join(folder, 'download.info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            info = get_necessary_info(json.load(f))
    else:
        info = {'title': os.path.basename(folder), 'uploader': 'Unknown', 'tags': []}
        
    transcript_path = os.path.join(folder, 'transcript.json')
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    
    summary_path = os.path.join(folder, 'summary.json')
    if os.path.exists(summary_path):
        summary = json.load(open(summary_path, 'r', encoding='utf-8'))
    else:
        summary = summarize(info, transcript, target_language, method)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    translation = _translate(summary, transcript, target_language, method)
    for i, line in enumerate(transcript):
        line['translation'] = translation[i]
    
    transcript = split_sentences(transcript)
    with open(os.path.join(folder, 'translation.json'), 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    return summary, transcript

def translate_all_transcript_under_folder(folder, method, target_language):
    s, t = None, None
    for root, dirs, files in os.walk(folder):
        if 'transcript.json' in files and 'translation.json' not in files:
            s, t = translate(method, root, target_language)
        elif 'translation.json' in files:
            s = json.load(open(os.path.join(root, 'summary.json'), 'r', encoding='utf-8'))
            t = json.load(open(os.path.join(root, 'translation.json'), 'r', encoding='utf-8'))
    return f'Processed {folder}', s, t
