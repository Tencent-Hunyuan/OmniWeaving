"""
Step4: Instruction Rewriting

Note: Ideally, different instruction rewriting prompts should be designed for
different edit meta-types (e.g., background replacement, object addition,
object replacement, style transfer, appearance modification) to produce more
accurate and natural rewritten instructions. 
Here we give a single unified prompt for simplicity and generality.

This script rewrites the original video editing instruction so that it
explicitly references the extracted element image (produced by step2 and
verified by step3). The rewritten instruction tells the model to use the
element shown in the provided image rather than a textual description.

A unified prompt is used that handles all edit types (background change,
object replacement, object addition, style change, etc.) without requiring
explicit type categorisation.

For each valid sample, the script:
  1. Reads the step3 verification output to find entries with at least one
     passing extracted image.
  2. Selects the best extracted image (first one that passed verification).
  3. Sends the extracted image together with the original instruction to a
     VLM to rewrite the instruction.
  4. Saves the rewritten instruction along with the selected image path.

Input (--step3_json):
    A JSONL file produced by step3 (extraction verification results).
    Each line is a JSON object containing:
        - videoid:               unique identifier for the video pair
        - condition_video_path:  path to the before-edit video
        - gt_video_path:         path to the after-edit video
        - condition_frame_path:  path to the first frame of the condition video
        - gt_frame_path:         path to the first frame of the gt video
        - new_element:           description of the newly introduced element
        - instruction:           the original edit instruction
        - extract_imgs:          list of dicts, each containing:
            - prompt_id:         extraction prompt variant index
            - extracted_path:    path to the extracted element image
            - extraction_result: "yes" / "no" — verification result
        - has_pass:              whether any extracted image passed verification
    Only entries with has_pass == True are processed.

Output (--output):
    A JSONL file where each line is a JSON object containing:
        - videoid:               unique identifier for the video pair
        - condition_video_path:  path to the before-edit video
        - gt_video_path:         path to the after-edit video
        - condition_frame_path:  path to the first frame of the condition video
        - gt_frame_path:         path to the first frame of the gt video
        - new_element:           description of the newly introduced element
        - instruction:           the original edit instruction
        - best_extracted_path:   path to the best extracted element image
        - best_prompt_id:        prompt variant index of the best extraction
        - rewritten:             rewritten instruction referencing the image
        - rewritten_ok:          "yes" / "no" — whether the rewrite is valid

    This file can be directly used as the --step4_json for step5.py.

Usage:
    python step4.py \\
        --step3_json /path/to/step3_output.jsonl \\
        --output /path/to/step4_result.jsonl \\
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

INSTRUCTION_REWRITE_PROMPT = '''You are an instruction rewriting expert. I will provide you with a video editing instruction and an image. The image contains the target element (e.g., a new background, a new object, or a modified subject) that will be used in the editing. Your task is to rewrite the original instruction so that it explicitly references the provided image.

Original Instruction: {instr}

### Requirements:
1. Determine whether the target element described in the original instruction corresponds to the main subject present in the provided image. If it does, rewrite the instruction to reference the image; if not, return an empty string, i.e., {{"rewritten": ""}}.
2. When rewriting, append "in the image" or "as shown in the image" after the target element to indicate it comes from the provided image. Remove detailed appearance modifiers of the target element from the original instruction, since the image already conveys that information. However, do NOT remove positional or action-related modifiers that describe where or how the element should be placed.
3. Strictly maintain the original intent and sentence structure of the instruction. Only modify the part that describes the target element.
4. Directly output the rewritten instruction in the "rewritten" field, without any additional explanation.

### Examples:

Example 1 (Background replacement):
Original Instruction: Replace background with a dynamic enchanted fairy forest featuring glowing mushrooms, floating lights, mist, and subtle shimmering lighting while keeping the subject still.
Output: {{"rewritten": "Replace the background with the scene shown in the image while keeping the subject still."}}
Example 2 (Object replacement):
Original Instruction: Replace the middle-aged man with a wise elderly man with silver hair and wrinkles.
Output: {{"rewritten": "Replace the middle-aged man with the elderly man in the image."}}

Example 3 (Object addition):
Original Instruction: Add the man with short, neatly combed gray hair wearing a white button-up shirt with a logo on the left chest, standing with his left hand on his hip, facing the woman who holds a microphone.
Output: {{"rewritten": "Add the man in the image to stand with his left hand on his hip and face the woman who holds a microphone."}}

Example 4 (Appearance modification - subject with transformation already applied in the image):
Original Instruction: Add a neatly trimmed dark beard to the man's face while keeping the pose and position.
Output: {{"rewritten": "Replace the man with the man having a trimmed dark beard in the image while keeping the pose and position."}}

Example 5 (Hairstyle change):
Original Instruction: Change the man's hair to vibrant platinum blonde, matching his head shape and pose.
Output: {{"rewritten": "Replace the man with the man with vibrant platinum blond hair in the image."}}

Example 6 (Clothing change):
Original Instruction: Replace the woman's blue dress with an elegant red evening gown.
Output: {{"rewritten": "Replace the woman's blue dress with the outfit shown in the image."}}

Example 7 (Style transfer):
Original Instruction: Transform the scene into a watercolor painting style with soft pastel colors.
Output: {{"rewritten": "Transform the scene into the artistic style shown in the image."}}

Example 8 (Object does not match the image):
Original Instruction: Add vibrant red headphones to the person.
(The image does not contain headphones)
Output: {{"rewritten": ""}}

### Output Format:
{{"rewritten": "..."}}
'''


# ────────────────────────── utilities ──────────────────────────

def clean_unicode_zeros(obj):
    """Recursively strip zero-width and other invisible Unicode characters."""
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


def encode_image(path: str, max_side=None) -> str:
    """Read an image file and return its base64-encoded string, optionally down-scaling."""
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

def send_rewrite_request(url: str, model_name: str, img_path: str,
                         prompt_text: str, max_new_tokens: int, timeout: int = 300):
    """Send an instruction rewriting request with one image to the VLM.

    Returns:
        (success, error_msg, rewritten_str) where rewritten_str is the
        rewritten instruction text (may be empty if the element doesn't match).
    """
    try:
        b64 = encode_image(img_path, max_side=640)
    except Exception as e:
        return False, f"Image encoding error: {e}", None

    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt_text},
            ]
        }],
        "max_tokens": max_new_tokens,
    }

    for attempt in range(6):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                parsed = extract_json_from_content(content)
                rewritten = parsed['rewritten']
                return True, "", rewritten
            else:
                print(f"Rewrite attempt {attempt + 1}: HTTP {resp.status_code}")
                if attempt >= 5:
                    return False, f"HTTP {resp.status_code}", None
        except Exception as e:
            print(f"Rewrite attempt {attempt + 1} failed: {e}")
            if attempt >= 5:
                return False, str(e), None

    return False, "Max retries reached", None


# ────────────────────────── data loading ──────────────────────────

def load_input_data(step3_json_path: str) -> list:
    """Load valid entries from the step3 output JSONL file.

    Keeps only entries where:
        - has_pass == True (at least one extracted image passed verification)
        - At least one extracted image with extraction_result == "yes" exists on disk

    For each valid entry, selects the best extracted image (first one that
    passed verification).
    """
    items = []
    with open(step3_json_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not entry.get('has_pass', False):
                continue

            # Select the best extracted image (first passing one)
            extract_imgs = entry.get('extract_imgs', [])
            best_img = None
            for img_info in extract_imgs:
                if img_info.get('extraction_result', '').lower() == 'yes':
                    img_path = img_info.get('extracted_path', '')
                    if img_path and os.path.exists(img_path):
                        best_img = img_info
                        break

            if best_img is None:
                continue

            items.append({
                'videoid': entry['videoid'],
                'condition_video_path': entry.get('condition_video_path', ''),
                'gt_video_path': entry.get('gt_video_path', ''),
                'condition_frame_path': entry.get('condition_frame_path', ''),
                'gt_frame_path': entry.get('gt_frame_path', ''),
                'new_element': entry.get('new_element', ''),
                'instruction': entry.get('instruction', ''),
                'best_extracted_path': best_img['extracted_path'],
                'best_prompt_id': best_img['prompt_id'],
            })

    return items


def load_existing_results(save_path: str) -> set:
    """Load already-processed videoids from the output file for resumption."""
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

def process_single_item(item, api_url, model_name, max_new_tokens):
    """Process a single entry: rewrite its instruction using the best extracted image.

    Returns:
        (result_dict, flag):
            flag:  1 = rewrite succeeded (non-empty),
                   0 = rewrite returned empty (element mismatch),
                  -1 = API error
    """
    instruction = item['instruction']
    best_img_path = item['best_extracted_path']

    prompt = INSTRUCTION_REWRITE_PROMPT.format(instr=instruction)

    ok, err, rewritten = send_rewrite_request(
        api_url, model_name, best_img_path, prompt, max_new_tokens
    )

    if not ok:
        return None, -1

    # Check if rewrite is valid
    is_valid = bool(rewritten and rewritten.strip() and rewritten.strip() != '...')

    result_dict = {
        'videoid': item['videoid'],
        'condition_video_path': item.get('condition_video_path', ''),
        'gt_video_path': item.get('gt_video_path', ''),
        'condition_frame_path': item['condition_frame_path'],
        'gt_frame_path': item['gt_frame_path'],
        'new_element': item['new_element'],
        'instruction': instruction,
        'best_extracted_path': best_img_path,
        'best_prompt_id': item['best_prompt_id'],
        'rewritten': rewritten,
        'rewritten_ok': 'yes' if is_valid else 'no',
    }

    return result_dict, 1 if is_valid else 0


def process_all(all_items, save_path, api_url, model_name, max_new_tokens):
    """Process all pending items and append results to the output JSONL file."""
    done_ids = load_existing_results(save_path)
    todo_items = [item for item in all_items if item['videoid'] not in done_ids]

    print(f'Total: {len(all_items)}, already done: {len(done_ids)}, todo: {len(todo_items)}')

    rewrite_ok = 0
    rewrite_empty = 0
    fail_count = 0

    for k, item in enumerate(tqdm(todo_items)):
        result_dict, flag = process_single_item(
            item, api_url, model_name, max_new_tokens
        )

        now_str = datetime.now().strftime("%H:%M:%S")

        if flag >= 0 and result_dict is not None:
            with open(save_path, 'a', encoding='utf8') as f:
                f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')

            if flag == 1:
                rewrite_ok += 1
            else:
                rewrite_empty += 1

            print(f'{now_str} [{k + 1}/{len(todo_items)}] '
                  f'instruction: {item["instruction"][:80]}... '
                  f'-> rewritten: {result_dict["rewritten"][:80]}... '
                  f'(ok={rewrite_ok}, empty={rewrite_empty}, fail={fail_count})')
        else:
            fail_count += 1
            print(f'{now_str} [{k + 1}/{len(todo_items)}] '
                  f'failed (total failures: {fail_count})')

    print(f'\nDone — rewrite_ok={rewrite_ok}, rewrite_empty={rewrite_empty}, '
          f'fail={fail_count}, total_processed={len(todo_items)}')


# ────────────────────────── entry point ──────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Step4: Instruction rewriting using VLM")
    p.add_argument("--step3_json", type=str, required=True,
                   help="Path to step3 output JSONL file (extraction verification results)")
    p.add_argument("--output", type=str, required=True,
                   help="Path to output JSONL file for saving rewritten instructions")
    p.add_argument("--server_ip", type=str, required=True,
                   help="VLM server IP address")
    p.add_argument("--server_port", type=str, default="8080",
                   help="VLM server port (default: 8080)")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-235B-A22B-Instruct",
                   help="Model name for the VLM API")
    p.add_argument("--max_new_tokens", type=int, default=4096,
                   help="Max new tokens for generation (default: 4096)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load input data
    all_items = load_input_data(args.step3_json)
    print(f'Loaded {len(all_items)} valid items from {args.step3_json}')

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    api_url = f'http://{args.server_ip}:{args.server_port}/v1/chat/completions'

    process_all(
        all_items=all_items,
        save_path=args.output,
        api_url=api_url,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == '__main__':
    main()
