"""
Step1: Object Identification and Self-Check

This script identifies the main moving objects in an image given a motion instruction,
and then performs a self-check to verify the identification results.

Input (--input_json):
    A JSON file containing training data as a list of dicts. Each dict must include:
        - videoid:    unique identifier for the video
        - video_path: path to the source video file
        - image_path: path to a frame extracted from the video
        - prompt:     text instruction describing the motion in the video

Output (--output):
    A JSONL file where each line is a JSON object with the identification results.

Usage:
    python step1.py \\
        --input_json /path/to/data.json \\
        --output /path/to/result.jsonl \\
        --server_ip vllm_ip \\
        --server_port vllm_port \\
        --model_name "Qwen/Qwen3-VL-235B-A22B-Instruct"
"""

import json
import os
import re
import io
import base64
import argparse
from datetime import datetime

import requests
from PIL import Image
from tqdm import tqdm


# ────────────────────────── prompts ──────────────────────────

PROMPT_WITH_REWRITE = '''
Given the image and the following text instruction that describes the subsequent motion in the image, you need to first identify the main objects within the image that will undergo motion and then rewrite the instruction based on the identified objects.

Instruction: {ins}

### Step1: Identifying the main objects that will undergo motion
Requirements:
1. Based on the motion instruction and the given image, identify the primary object(s) that will move or change in the subsequent action. Return a description of the corresponding object(s) **using a phrase of a few words** in the following dictionary format: {{'[object1]': '...'}}.
2. If there is more than one primary object undergoing motion, all of these objects must be returned. Use the following format: {{'[object1]': '...', '[object2]': '...'}} or {{'[object1]': '...', '[object2]': '...', '[object3]': '...'}}. Note that you should only identify the major moving objects, and the number of returned objects must NOT exceed three.
3. If the image contains multiple objects of the same category as the intended subject, or if the multiple objects undergoing motion belong to the same category, you must provide a clear and unambiguous description for each [object] that allows a person to precisely locate the specific object in the image. For example, if there are several men in the picture, and the instruction requires two specific men to move, [object1] and [object2] must clearly describe which man they correspond to, rather than simply using the generic term 'man'.
4. If there's no ambiguity in reference, describe the object with as few words as possible, such as 1-2 words. For example, if there's only one man in the image, the single word 'man' is sufficient. Do not add any modifiers such as 'man in a yellow hat'.
5. The object you identify must be a complete entity, **NOT a partial component or body part of a subject** (e.g., a bird's wing or a person's arms or a person's mouth are body parts/organs and should NOT be returned).
6. If you identify multiple objects (e.g., [object1]+[object2] or [object1]+[object2]+[object3]), the description for each individual object must NOT include any reference to the other objects. For example, the description of [object1] must not contain any reference to [object2].
7. Do not select the object that is related to the background of the image. Do not select the larger setting in which the image takes place as the object.
8. The prompt may contain descriptions related to camera motion. When identifying the main objects in the image that will move, disregard these descriptions of camera changes.
9. If the given instruction and the image do not correspond, return an empty dictionary. If the given instruction and the image do not correspond, return an empty dictionary. If you cannot infer any main object that will undergo motion, return an empty dictionary.
10. Verify if the object you identify actually exists within the image. If not, return an empty dictionary.
11. Directly output the description of the corresponding objects without any extra explanation.

### Step2: Instruction rewarting
Requirements:
1. Based on the main objects you identified in Step 1, rewrite the original motion instruction without altering its original semantic meaning. During rewriting, you only need to replace the object(s) that will undergo motion with the symbolic indicators: [object1], [object2], or [object3]. **Do NOT replace them with the actual object descriptions you generated**.
2. Some instructions may use numerical quantifiers to simultaneously indicate multiple moving objects (e.g., "three men are running," corresponding to [object1], [object2], and [object3]). In the rewritten instruction, you should remove the numerical quantifier and replace the subjects with the list of symbolic indicators (e.g., [object1], [object2], and [object3]).
3. Besides the above two points, Strictly maintain the original content and sentence structure of the instruction. What you only need to do is replacing the object(s) that will undergo motion with the symbolic indicators.
4. If you do not identify any object in the first step, return the original instruction without rewriting. Otherwise, make sure to conduct the replacement and do NOT return the original instruction.
5. Directly output the rewritten instruction without any extra explanation.

### Output Format if you do not identify any object that will undergo motion
{{"moving_objects":{{}}, "rewritten_instruction":"..."}}

### Output Format if you identify one object that will undergo motion
{{"moving_objects":{{"[object1]": "..."}}, "rewritten_instruction":"..."}}

### Output Format if you identify two objects that will undergo motion
{{"moving_objects":{{"[object1]": "...", "[object2]": "..."}}, "rewritten_instruction":"..."}}

### Output Format if you identify three objects that will undergo motion
{{"moving_objects":{{"[object1]": "...", "[object2]": "...", "[object3]": "..."}}, "rewritten_instruction":"..."}}
'''

PROMPT_WITHOUT_REWRITE = '''
Given the image and the following text instruction that describes the subsequent motion in the image, you need to first identify the main objects within the image that will undergo motion.

Instruction: {ins}

Requirements:
1. Based on the motion instruction and the given image, identify the primary object(s) that will move or change in the subsequent action. Return a description of the corresponding object(s) **using a phrase of a few words** in the following dictionary format: {{'[object1]': '...'}}.
2. If there is more than one primary object undergoing motion, all of these objects must be returned. Use the following format: {{'[object1]': '...', '[object2]': '...'}} or {{'[object1]': '...', '[object2]': '...', '[object3]': '...'}}. Note that you should only identify the major moving objects, and the number of returned objects must NOT exceed three.
3. If the image contains multiple objects of the same category as the intended subject, or if the multiple objects undergoing motion belong to the same category, you must provide a clear and unambiguous description for each [object] that allows a person to precisely locate the specific object in the image. For example, if there are several men in the picture, and the instruction requires two specific men to move, [object1] and [object2] must clearly describe which man they correspond to, rather than simply using the generic term 'man'.
4. If there's no ambiguity in reference, describe the object with as few words as possible, such as 1-2 words. For example, if there's only one man in the image, the single word 'man' is sufficient. Do not add any modifiers such as 'man in a yellow hat'.
5. The object you identify must be a complete entity, **NOT a partial component or body part of a subject** (e.g., a bird's wing or a person's arms or a person's mouth are body parts/organs and should NOT be returned).
6. If you identify multiple objects (e.g., [object1]+[object2] or [object1]+[object2]+[object3]), the description for each individual object must NOT include any reference to the other objects. For example, the description of [object1] must not contain any reference to [object2].
7. Do not select the object that is related to the background of the image. Do not select the larger setting in which the image takes place as the object.
8. The prompt may contain descriptions related to camera motion. When identifying the main objects in the image that will move, disregard these descriptions of camera changes.
9. If the given instruction and the image do not correspond, return an empty dictionary. If the given instruction and the image do not correspond, return an empty dictionary. If you cannot infer any main object that will undergo motion, return an empty dictionary.
10. Verify if the object you identify actually exists within the image. If not, return an empty dictionary.
11. Directly output the description of the corresponding objects without any extra explanation.

### Output Format if you do not identify any object that will undergo motion
{{"moving_objects":{{}}}}

### Output Format if you identify one object that will undergo motion
{{"moving_objects":{{"[object1]": "..."}}}}

### Output Format if you identify two objects that will undergo motion
{{"moving_objects":{{"[object1]": "...", "[object2]": "..."}}}}

### Output Format if you identify three objects that will undergo motion
{{"moving_objects":{{"[object1]": "...", "[object2]": "...", "[object3]": "..."}}}}
'''

PROMPT_CHECK_SINGLE = '''
Given the image, the instruction that describes the subsequent motion in the image, and the object description that describes the objects in the image, you need to make the following judgments regarding the content of the object description.

Instruction: {ins}
Object description: {desc}

1. The described object exists in the image.
2. The described object has been mentioned in the instruction. Note that the description of the object in the instruction does NOT need to be consistent with the Object description.
3. The object description must be a complete entity, NOT a partial component or body part of a subject.

If the input object description fully meets the above four requirements, return yes, otherwise return no and tell me the reason.

### Output Format:
{{"result":"yes/no", "reason":"..."}}
'''

PROMPT_CHECK_MULTI = '''
Given the image, the instruction that describes the subsequent motion in the image, and the object description that describes the main objects in the image, you need to make the following judgments regarding the content of the object description.

Instruction: {ins}
Object description: {desc}

For each object mentioned in the Object description, it must meet the following four criteria:
1. The described object exists in the image.
2. The described object has been mentioned in the instruction. (Note: The description of the object in the instruction does NOT need to be consistent with the Object description.)
3. The object description must refer to a complete entity, NOT a partial component or a body part of a subject. For example, the object description can NOT refer to a person's arm or hand.
4. The object description can NOT contain any reference to {others} in the image.

If all described objects fully meet the above four requirements, return 'yes', otherwise return 'no' and provide the reason.

### Output Format:
{{"result":"yes/no", "reason":"..."}}
'''

VALID_OBJECT_KEYS = {"[object1]", "[object2]", "[object3]"}


# ────────────────────────── utilities ──────────────────────────

def clean_unicode_zeros(obj):
    """Recursively strip zero-width and other invisible Unicode characters from a JSON object."""
    zero_width_chars = ('\u200b', '\u200c', '\u200d', '\ufeff', '\u200e', '\u200f')
    if isinstance(obj, dict):
        return {k: clean_unicode_zeros(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_unicode_zeros(item) for item in obj]
    elif isinstance(obj, str):
        for ch in zero_width_chars:
            obj = obj.replace(ch, '')
        return obj
    return obj


def extract_json_from_content(content_str: str):
    """Extract and parse JSON content from the model's response string."""
    if not content_str:
        return None

    content_str = content_str.strip()

    # Try to match ```json ... ``` or ``` ... ``` fenced code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content_str, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        start_idx = content_str.find('{')
        end_idx = content_str.rfind('}')
        if start_idx != -1 and end_idx > start_idx:
            json_str = content_str[start_idx:end_idx + 1]
        else:
            json_str = content_str

    # Remove redundant blank lines and zero-width characters
    json_str = re.sub(r'\n\s*\n+', '\n', json_str)
    for ch in ('\u200b', '\u200c', '\u200d', '\ufeff'):
        json_str = json_str.replace(ch, '')
    json_str = json_str.strip()

    return clean_unicode_zeros(json.loads(json_str))


def decode_image(path: str, max_side=None) -> str:
    """Read an image file and return its base64-encoded string, optionally rescaling by the longest side."""
    im = Image.open(path).convert("RGB")
    if max_side and max_side > 0:
        w, h = im.size
        scale = max(w, h) / float(max_side)
        if scale > 1:
            im = im.resize((int(round(w / scale)), int(round(h / scale))), Image.BICUBIC)

    buffer = io.BytesIO()
    im.save(buffer, format='JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


# ────────────────────────── API helpers ──────────────────────────

def _build_payload(model_name: str, image_b64: str, prompt_text: str, max_new_tokens: int) -> dict:
    """Build the VLM API request payload."""
    return {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": prompt_text},
            ]
        }],
        "max_tokens": max_new_tokens,
    }


def send_identify_request(url: str, model_name: str, image_b64: str, prompt_text: str,
                          max_new_tokens: int, timeout: int = 30):
    """Send an object identification request. Returns (success, error_msg, content)."""
    payload = _build_payload(model_name, image_b64, prompt_text, max_new_tokens)
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                print('str:', content)
                content = extract_json_from_content(content)
                return True, "", content
            else:
                print(f"HTTP {resp.status_code}: {resp.text}")
                if attempt >= 2:
                    return False, f"HTTP {resp.status_code}: {resp.text}", None
        except Exception as e:
            print(e)
            if attempt >= 2:
                return False, str(e), None
    return False, "", None


def send_check_request(url: str, model_name: str, image_b64: str, prompt_text: str,
                       max_new_tokens: int, timeout: int = 30):
    """Send an object verification (self-check) request. Returns (success, error_msg, content)."""
    payload = _build_payload(model_name, image_b64, prompt_text, max_new_tokens)
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                content = extract_json_from_content(content)
                if content["result"].lower() in ('yes', 'no'):
                    return True, "", content
            else:
                print(f"HTTP {resp.status_code}: {resp.text}")
                if attempt >= 2:
                    return False, f"HTTP {resp.status_code}: {resp.text}", None
        except Exception as e:
            print(e)
            if attempt >= 2:
                return False, str(e), None
    return False, "", None


# ────────────────────────── data loading ──────────────────────────

def load_input_data(input_json_path: str) -> list:
    """
    Load the data list from the input JSON file.

    The JSON file should be a list of dicts, each containing training data fields.
    Required fields per entry: videoid, video_path, image_path, prompt.
    """
    with open(input_json_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    items = []
    for idx, entry in enumerate(data):
        videoid = entry.get('videoid', '')

        caption = entry.get('prompt', '')

        image_path = entry.get('image_path', '')

        if not videoid or not caption or not image_path:
            continue
        if not os.path.exists(image_path):
            continue

        items.append({
            'videoid': videoid,
            'image_path': image_path,
            'index': idx,
            'caption': caption,
        })

    return items


def load_existing_results(save_path: str) -> set:
    """Load the set of already-processed videoids from the output file to skip them."""
    if not os.path.exists(save_path):
        return set()

    done_ids = set()
    with open(save_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                done_ids.add(item["videoid"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done_ids


# ────────────────────────── main processing ──────────────────────────

def process_single_item(md, api_url, model_name, max_new_tokens, max_time):
    """Process a single item: identify objects -> self-check -> return results.

    Returns (json_save, flag):
        flag:  1 = success, 0 = empty objects, 2 = check failed, -1 = error
    """
    try:
        imgb64 = decode_image(md['image_path'], 640)
    except Exception:
        return None, -1

    instr_en = md['caption']
    json_save = None
    flag = -1

    for cur_time in range(1, max_time + 1):
        # First attempt includes rewrite; subsequent attempts do not
        if cur_time <= 1:
            cur_prompt = PROMPT_WITH_REWRITE.format(ins=instr_en)
        else:
            cur_prompt = PROMPT_WITHOUT_REWRITE.format(ins=instr_en)

        ok, err, content = send_identify_request(api_url, model_name, imgb64, cur_prompt, max_new_tokens)

        if not ok:
            flag = -1
            continue

        json_save = {
            'index': md['index'],
            'image_name': md['image_path'],
            'videoid': md['videoid'],
            'instruction': instr_en,
            'objects': content,
        }

        # Parse moving_objects
        try:
            objects = content['moving_objects']
        except (KeyError, TypeError):
            json_save["overall"] = "no"
            flag = -1
            continue

        if len(objects) == 0:
            json_save["overall"] = "no"
            flag = 0
            break

        # Validate object keys
        obj_names = []
        valid = True
        for obj_key, obj_desc in objects.items():
            if obj_key in VALID_OBJECT_KEYS:
                obj_names.append((obj_key, obj_desc))
            else:
                valid = False
                break

        if not valid:
            json_save["overall"] = "no"
            flag = -1
            continue

        # self-check
        check_results = {}
        check_flag = 1
        for obj_idx, obj_name in obj_names:
            if len(obj_names) == 1:
                check_prompt = PROMPT_CHECK_SINGLE.format(ins=instr_en, desc=obj_name)
            else:
                others_parts = [ccc for _, ccc in obj_names if ccc != obj_name]
                others = ', or '.join(others_parts)
                check_prompt = PROMPT_CHECK_MULTI.format(ins=instr_en, desc=obj_name, others=others)

            c_ok, c_err, c_result = send_check_request(api_url, model_name, imgb64, check_prompt, max_new_tokens)
            if c_ok:
                check_results[obj_idx] = c_result
                if c_result['result'].lower() != 'yes' and check_flag != -1:
                    check_flag = 2
            else:
                check_flag = -1

        json_save["selfcheck"] = check_results

        if check_flag == 1:
            json_save["overall"] = "yes"
            flag = 1
            print(f'  try {cur_time}: success')
            break
        else:
            json_save["overall"] = "no"
            flag = check_flag
            print(f'  try {cur_time}: fail (flag={check_flag})',
                  json_save['objects'].get('moving_objects', {}))

        if cur_time == max_time:
            print('  skip this case ~ max retries reached')

    return json_save, flag


def process_all(all_items, save_path, api_url, model_name, max_new_tokens, max_time):
    """Process all pending items."""
    # Load already-completed entries to support resumption
    done_ids = load_existing_results(save_path)
    todo_items = [item for item in all_items if item['videoid'] not in done_ids]

    print(f'Total: {len(all_items)}, already done: {len(done_ids)}, todo: {len(todo_items)}')

    correct = 0
    wrong_check = 0
    wrong_error = 0

    for k, md in enumerate(tqdm(todo_items)):
        json_save, flag = process_single_item(md, api_url, model_name, max_new_tokens, max_time)

        now_str = datetime.now().strftime("%H:%M:%S")
        total = correct + wrong_check + wrong_error

        if flag != -1 and json_save is not None:
            with open(save_path, 'a', encoding='utf8') as f:
                f.write(json.dumps(json_save, ensure_ascii=False) + '\n')

            if flag == 1:
                correct += 1
            else:
                wrong_check += 1

            print(f'{now_str} {k}/{len(todo_items)} '
                  f'({total + 1}={correct + (1 if flag == 1 else 0)}'
                  f'+{wrong_check + (1 if flag != 1 else 0)}+{wrong_error}): '
                  f'{json_save["objects"].get("moving_objects", {})} {json_save["overall"]}')
        else:
            wrong_error += 1
            print(f'{now_str} {k}/{len(todo_items)} '
                  f'({total + 1}={correct}+{wrong_check}+{wrong_error}): error/skip')

    print(f'\nDone — correct={correct}, wrong_check={wrong_check}, wrong_error={wrong_error}')


# ────────────────────────── entry point ──────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Step1: Object identification and self-check")
    p.add_argument("--input_json", type=str, required=True,
                   help="Path to input JSON file containing training data. "
                        "Must be a list of dicts, each with: videoid, video_path, "
                        "image_path (a frame extracted from the video), and prompt")
    p.add_argument("--output", type=str, required=True,
                   help="Path to output JSONL file for saving results")
    p.add_argument("--server_ip", type=str, required=True, help="Server IP address")
    p.add_argument("--server_port", type=str, default="8080", help="Server port (default: 8080)")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-235B-A22B-Instruct",
                   help="Model name for the VLM API")
    p.add_argument("--max_time", type=int, default=2, help="Max retry times per sample")
    p.add_argument("--max_new_tokens", type=int, default=4096, help="Max new tokens for generation")
    return p.parse_args()


def main():
    args = parse_args()

    # Load input data
    all_items = load_input_data(args.input_json)
    print(f'Loaded {len(all_items)} valid items from {args.input_json}')

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    api_url = f'http://{args.server_ip}:{args.server_port}/v1/chat/completions'

    process_all(
        all_items=all_items,
        save_path=args.output,
        api_url=api_url,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        max_time=args.max_time,
    )


if __name__ == '__main__':
    main()
