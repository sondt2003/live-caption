import os
import re
import json
import torch
from loguru import logger
from .factory import TranslatorFactory

def get_transcript_summary(transcript):
    """
    Get a summary of the transcript for context-aware translation.
    """
    all_text = " ".join([seg['text'] for seg in transcript])
    return all_text[:500] # Limit to 500 chars for summary context

def repair_json(s):
    if not s:
        return s
    
    s = s.strip()
    
    # Remove markdown code blocks if present
    if "```json" in s:
        s = s.split("```json")[1].split("```")[0].strip()
    elif "```" in s:
        s = s.split("```")[1].split("```")[0].strip()
    
    # If it still doesn't look like JSON, try to extract it
    if not (s.startswith('{') or s.startswith('[')):
        start_brace = s.find('{')
        start_bracket = s.find('[')
        
        if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
            s = s[start_brace : s.rfind('}') + 1]
        elif start_bracket != -1:
            s = s[start_bracket : s.rfind(']') + 1]

    # Try to fix unescaped quotes between property names and separators
    def replace_quotes(match):
        prefix = match.group(1)
        content = match.group(2)
        suffix = match.group(3)
        # Escape internal quotes
        content = content.replace('"', '\\"')
        return f'{prefix}"{content}"{suffix}'

    pattern = r'(:[ \n]*)"(.*?)"([ \n]*[,}])'
    s = re.sub(pattern, replace_quotes, s, flags=re.DOTALL)
    
    return s

def split_text_into_sentences(para):
    # Support both English and CJK punctuation
    # Use negative lookahead (?![0-9]) to avoid splitting at decimal points or thousand separators
    para = re.sub(r'([。！？\?\.\!])(?![0-9])([^，。！？\?\.\!”’》])', r"\1\n\2", para)
    para = re.sub(r'(\.{6})([^，。！？\?\.\!”’》])', r"\1\n\2", para)
    para = re.sub(r'(\…{2})([^，。！？\?\.\!”’》])', r"\1\n\2", para)
    para = re.sub(r'([。！？\?\.\!][”’])([^，。！？\?\.\!”’》])', r'\1\n\2', para)
    para = para.rstrip()
    sentences = para.split("\n")
    
    # Fallback for very long segments without any punctuation (e.g. more than 30 words)
    final_sentences = []
    for s in sentences:
        words = s.split()
        if len(words) > 30:
            for i in range(0, len(words), 20):
                final_sentences.append(" ".join(words[i:i+20]))
        else:
            final_sentences.append(s)
            
    return [s.strip() for s in final_sentences if s.strip()]

def _translate(summary, transcript, target_language, method):
    translator = TranslatorFactory.get_translator(method, target_language)
    
    # Batching logic: Groq/LLMs work better with context
    # We send max 1000 chars (approx 200-300 words), unlimited segments
    batches = []
    curr_batch = []
    curr_chars = 0
    
    # Batching logic: Groq/LLMs work better with context, Google works better with large blocks
    is_traditional = method.lower() in ['google', 'bing']
    # Traditional engines are sensitive to total character count including tags
    # and number of segments. 1000 chars and 50 segments is a very safe limit.
    max_chars = 1000 if is_traditional else 500
    max_segments = 50 if is_traditional else 100
    
    batches = []
    curr_batch = []
    curr_chars = 0
    tag_overhead = 20 # Estimate for <p id="123">...</p>
    
    for i, seg in enumerate(transcript):
        text = seg['text']
        seg_cost = len(text) + (tag_overhead if is_traditional else 0)
        
        if curr_batch and (curr_chars + seg_cost > max_chars or len(curr_batch) >= max_segments):
            batches.append(curr_batch)
            curr_batch = []
            curr_chars = 0
        curr_batch.append({"id": i, "text": text})
        curr_chars += seg_cost
    if curr_batch:
        batches.append(curr_batch)
        
    all_translations = [None] * len(transcript)
    
    for batch in batches:
        # Traditional translators (Google/Bing) don't handle JSON well
        is_json_method = method.lower() not in ['google', 'bing']
        
        if is_json_method:
            system_prompt = f"Translate the following subtitles to {target_language}. Context: {summary}. \n" \
                            f"Rules:\n" \
                            f"1. Use natural spoken Vietnamese (I='mình', You='các bạn').\n" \
                            f"2. Return STRICT JSON: {{ \"id\": \"translation\" }}. Escape all double quotes inside the translation text with backslashes.\n" \
                            f"3. Maintain original meaning but make it sound like a Vlog.\n" \
                            f"4. Do not include any text other than the JSON object."
            
            user_content = json.dumps(batch, ensure_ascii=False)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            response_json = translator.translate(messages)
        else:
            # Strategy for Google/Bing: HTML tags for max stability
            # Google Translate preserves HTML structure and attributes perfectly
            lines = [f'<p id="{item["id"]}">{item["text"]}</p>' for item in batch]
            user_content = "".join(lines)
            messages = [{"role": "user", "content": user_content}]
            
            # Use json_mode=False for direct text processing
            response_text = translator.translate(messages, json_mode=False)
            
            if response_text:
                temp_trans = {}
                # Regex to extract id and content from <p id="idx">content</p>
                # Case-insensitive and handles single/double/no quotes for id attribute
                matches = re.findall(r'<p id=["\']?(\d+)["\']?>(.*?)</p>', response_text, re.DOTALL | re.IGNORECASE)
                for it_id, it_text in matches:
                    try:
                        temp_trans[int(it_id)] = it_text.strip()
                    except:
                        continue
                
                # Check if we got enough translations back
                if len(temp_trans) >= len(batch) * 0.7:
                    for item in batch:
                        idx = item['id']
                        all_translations[idx] = temp_trans.get(idx, item['text'])
                    continue # Batch handled, move to next
                else:
                    logger.warning(f"Batch translation failed for {method} (HTML count mismatch: {len(temp_trans)}/{len(batch)}). Falling back to individual.")
                    response_json = None
            else:
                response_json = None

        if response_json:
            try:
                # Clean and repair response
                response_json = repair_json(response_json)
                
                batch_trans = json.loads(response_json, strict=False)
                if isinstance(batch_trans, list):
                    # Handle Google/Bing returning a list of dicts [{"id": 0, "text": "trans"}, ...]
                    for item_trans in batch_trans:
                        it_id = item_trans.get('id')
                        it_text = item_trans.get('text')
                        if it_id is not None:
                            all_translations[int(it_id)] = it_text
                else:
                    # Handle LLM returning a dict { "0": "trans", "1": "trans", ... }
                    for item in batch:
                        idx = item['id']
                        all_translations[idx] = batch_trans.get(str(idx), item['text'])
            except Exception as e:
                logger.warning(f"Failed to parse batch json. Retrying individually... Error: {e}")
                # Fallback: Translate individually
                logger.warning(f"Batch translation failed for {len(batch)} items. Falling back to individual translation (Text Mode)...")
                for item in batch:
                    try:
                        s_prompt = f"Translate the following text to {target_language}. Context: {summary}. Return ONLY the translation text."
                        msgs = [{"role": "system", "content": s_prompt}, {"role": "user", "content": item['text']}]
                        # Use json_mode=False for better adherence on small models for single lines
                        res = translator.translate(msgs, json_mode=False)
                        all_translations[item['id']] = res if res else item['text']
                    except Exception as e:
                         all_translations[item['id']] = item['text']
        else:
            logger.warning(f"Translator returned None for batch of {len(batch)} items. Retrying individually (Text Mode)...")
            for item in batch:
                try:
                    s_prompt = f"Translate the following text to {target_language}. Context: {summary}. Return ONLY the translation text."
                    msgs = [{"role": "system", "content": s_prompt}, {"role": "user", "content": item['text']}]
                    res = translator.translate(msgs, json_mode=False)
                    all_translations[item['id']] = res if res else item['text']
                except:
                    all_translations[item['id']] = item['text']
                
    return all_translations

def split_sentences(transcript):
    new_transcript = []
    for line in transcript:
        sentences = split_text_into_sentences(line['translation'])
        if len(sentences) > 1:
            total_chars = len(line['translation'])
            curr_start = line['start']
            duration = line['end'] - line['start']
            for sent in sentences:
                sent_dur = (len(sent) / total_chars) * duration
                new_transcript.append({
                    'start': round(curr_start, 3),
                    'end': round(curr_start + sent_dur, 3),
                    'text': line['text'],
                    'translation': sent,
                    'speaker': line.get('speaker', 'SPEAKER_00')
                })
                curr_start += sent_dur
        else:
            new_transcript.append(line)
    return new_transcript

def translate(method, folder, target_language='vi'):
    transcript_path = os.path.join(folder, 'transcript.json')
    if not os.path.exists(transcript_path):
        return None, None
        
    transcript = json.load(open(transcript_path, 'r', encoding='utf-8'))
    summary = get_transcript_summary(transcript)
    
    with open(os.path.join(folder, 'summary.json'), 'w', encoding='utf-8') as f:
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
