"""
Step2: Instruction Rewriting and Self-Check

This script takes the output of step1 (object identification results) and rewrites
the original motion instruction by replacing object references with symbolic
indicators ([object1], [object2], [object3]). It then performs a self-check to
verify the rewriting quality.

Input (--input_json):
    A JSONL file produced by step1, where each line is a JSON object containing:
        - videoid:      unique identifier for the video
        - image_name:   path to the frame image
        - instruction:  original motion instruction
        - objects:      dict with "moving_objects" and optionally "rewritten_instruction"
        - overall:      "yes" / "no" from step1's self-check
    Only entries with overall == "yes" and non-empty moving_objects are processed.

Output (--output):
    A JSONL file where each line is a JSON object with the rewriting results,
    including the rewritten instruction and self-check verdict.

Usage:
    python step2.py \\
        --input_json /path/to/step1_output.jsonl \\
        --output /path/to/step2_result.jsonl \\
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

PROMPT_REWRITE = '''
Given the image, the instruction that describes the subsequent motion in the image, and the object description that describes the objects that will undergo motion in the image, you need to rewrite the instruction based on the object description.

### Requirements:
1. Establish a correspondence between the described object(s) and the instruction based on the image. And then rewrite the original instruction without altering its original semantic meaning. During rewriting, you only need to replace the object reference(s) in the instruction with the corresponding symbolic indicators: [object1], [object2], or [object3]. **Do NOT replace them with the actual object descriptions**.
2. Some instructions may use numerical quantifiers to simultaneously indicate multiple moving objects (e.g., "three men are running," corresponding to [object1], [object2], and [object3]). In the rewritten instruction, you should remove the numerical quantifier and replace the subjects with the list of symbolic indicators (e.g., [object1], [object2], and [object3]).
3. If the instruction contains Chinese, translate the Chinese into English, ensuring fluency and maintaining the original semantic meaning and sentence structure.
3. Besides the above three points, do not change any other content in the instruction. Strictly Maintain the original sentence structure of the instruction.
4. Directly output the rewritten instruction without any extra explanation.

### Examples:
Instruction example: A man stands holding a red cap before moving to the right and walking out of the frame entirely. A second person lies motionless on the ground beside a truck after initially being covered up.
Object description example: {{"[object1]": "the man holding red cap", "[object2]": "the person lying on ground"}}
Output example: {{"rewritten_instruction": "[object1] stands holding a red cap before moving to the right and walking out of the frame entirely. [object2] lies motionless on the ground beside a truck after initially being covered up."}}

### Input:
Instruction: {ins}
Object description: {desc}

### Output Format
{{"rewritten_instruction":"..."}}
'''

PROMPT_CHECK_REWRITE = '''
Given the image, the instruction that describes the subsequent motion in the image, and the object description that describes the objects that will undergo motion in the image, and the rewritten instruction, you need to determine whether the rewritten instruction has fulfilled the requirements of the rewriting.

Original Instruction: {ins}
Object Description: {desc}

### Rewriting Requirements:
1. Establish a correspondence between the described object(s) and the instruction based on the image. And then rewrite the original instruction without altering its original semantic meaning. During rewriting, you only need to replace the object reference(s) in the instruction with the corresponding symbolic indicators: [object1], [object2], or [object3]. **Do NOT replace them with the actual object descriptions**.
2. Some instructions may use numerical quantifiers to simultaneously indicate multiple moving objects (e.g., "three men are running," corresponding to [object1], [object2], and [object3]). In the rewritten instruction, you should remove the numerical quantifier and replace the subjects with the list of symbolic indicators (e.g., [object1], [object2], and [object3]).
3. If the instruction contains Chinese, translate the Chinese into English, ensuring fluency and maintaining the original semantic meaning and sentence structure.
4. Besides the above three points, do not change any other semantic meanings in the instruction. Also, maintain the original sentence structure of the instruction.

Rewritten Instruction: {newins}

If the rewritten instruction meets the above requirements, return yes, otherwise return no and tell me the reason.

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


def send_rewrite_request(url: str, model_name: str, image_b64: str, prompt_text: str,
                         max_new_tokens: int, timeout: int = 30):
    """Send an instruction rewriting request. Returns (success, error_msg, rewritten_instruction)."""
    payload = _build_payload(model_name, image_b64, prompt_text, max_new_tokens)
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                parsed = extract_json_from_content(content)
                rewritten = parsed["rewritten_instruction"]
                return True, "", rewritten
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
    """Send a rewrite verification (self-check) request. Returns (success, error_msg, content)."""
    payload = _build_payload(model_name, image_b64, prompt_text, max_new_tokens)
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                parsed = extract_json_from_content(content)
                if parsed["result"].lower() in ('yes', 'no'):
                    return True, "", parsed
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
    Load valid entries from the step1 output JSONL file.

    Only entries with overall == "yes" and non-empty, valid moving_objects are kept.
    Returns a list of dicts ready for rewriting.
    """
    items = []
    with open(input_json_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Only process entries that passed step1's self-check
            if entry.get("overall", "").lower() != "yes":
                continue

            image_path = entry.get('image_name', '')
            if not image_path or not os.path.exists(image_path):
                continue

            # Validate moving_objects
            try:
                objects = entry['objects']['moving_objects']
            except (KeyError, TypeError):
                continue
            if not objects or len(objects) == 0:
                continue

            # Validate all object keys
            if not all(k in VALID_OBJECT_KEYS for k in objects.keys()):
                continue

            items.append(entry)

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
    """Process a single item: rewrite instruction -> self-check -> return results.

    Returns (md, flag):
        flag:  1 = success, 0 = check failed, -1 = error
    """
    try:
        imgb64 = decode_image(md['image_name'], 640)
    except Exception:
        return md, -1

    instruction = md["instruction"]
    objects = md['objects']['moving_objects']
    obj_str = json.dumps(objects)
    obj_keys = sorted(objects.keys())

    # Use the rewritten_instruction from step1 as initial candidate (if available)
    rewritten = md['objects'].get("rewritten_instruction", None)
    flag = -1

    for cur_time in range(1, max_time + 1):
        flag = 1

        # Re-request rewriting if no candidate or on retry
        if cur_time > 1 or not rewritten or rewritten == '...':
            rewrite_prompt = PROMPT_REWRITE.format(ins=instruction, desc=obj_str)
            ok, err, result = send_rewrite_request(
                api_url, model_name, imgb64, rewrite_prompt, max_new_tokens)
            if ok:
                rewritten = result
            else:
                flag = -1
                rewritten = None

        md["rewritten"] = rewritten

        if not rewritten:
            flag = -1
            continue

        # Verify all object keys appear in the rewritten instruction
        for obj_key in obj_keys:
            if obj_key not in rewritten:
                flag = 0
                md['rewritten_selfcheck'] = {'result': 'no', 'reason': f'{obj_key} missing in rewritten instruction'}
                print(f'\n  rewrite error: {obj_key} not found in rewritten instruction\n')
                break

        # Send self-check request if keys are valid
        if flag == 1:
            check_prompt = PROMPT_CHECK_REWRITE.format(ins=instruction, desc=obj_str, newins=rewritten)
            c_ok, c_err, c_result = send_check_request(
                api_url, model_name, imgb64, check_prompt, max_new_tokens)
            if c_ok:
                md['rewritten_selfcheck'] = c_result
                if c_result.get('result', 'no').lower() != 'yes':
                    flag = 0
            else:
                flag = -1

        if flag == 1:
            print(f'  try {cur_time}: success')
            break
        else:
            print(f'  try {cur_time}: fail')

        if cur_time == max_time:
            print('  skip this case ~ max retries reached')

    return md, flag


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
        md, flag = process_single_item(md, api_url, model_name, max_new_tokens, max_time)

        now_str = datetime.now().strftime("%H:%M:%S")
        total = correct + wrong_check + wrong_error
        rewritten = md.get("rewritten", None)

        if flag != -1:
            with open(save_path, 'a', encoding='utf8') as f:
                f.write(json.dumps(md, ensure_ascii=False) + '\n')

            if flag == 1:
                correct += 1
            else:
                wrong_check += 1

            print(f'{now_str} {k}/{len(todo_items)} '
                  f'({total + 1}={correct + (1 if flag == 1 else 0)}'
                  f'+{wrong_check + (1 if flag != 1 else 0)}+{wrong_error}): '
                  f'{rewritten}')
        else:
            wrong_error += 1
            print(f'{now_str} {k}/{len(todo_items)} '
                  f'({total + 1}={correct}+{wrong_check}+{wrong_error}): error/skip')

    print(f'\nDone — correct={correct}, wrong_check={wrong_check}, wrong_error={wrong_error}')


# ────────────────────────── entry point ──────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Step2: Instruction rewriting and self-check")
    p.add_argument("--input_json", type=str, required=True,
                   help="Path to the step1 output JSONL file. Each line is a JSON object "
                        "with videoid, image_name, instruction, objects (including "
                        "moving_objects), and overall fields")
    p.add_argument("--output", type=str, required=True,
                   help="Path to output JSONL file for saving rewriting results")
    p.add_argument("--server_ip", type=str, required=True, help="VLM server IP address")
    p.add_argument("--server_port", type=str, default="8080", help="VLM server port (default: 8080)")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-235B-A22B-Instruct",
                   help="Model name for the VLM API")
    p.add_argument("--max_time", type=int, default=2, help="Max retry times per sample")
    p.add_argument("--max_new_tokens", type=int, default=4096, help="Max new tokens for generation")
    return p.parse_args()


def main():
    args = parse_args()

    # Load input data from step1 output
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
