"""
Step3: Extraction Quality Verification

Note: Ideally, different VLM prompts should be designed for different edit
meta-types (e.g., background replacement, object addition, object replacement,
style transfer) to achieve more accurate and targeted extraction verification.
Here we give a single unified prompt for simplicity and generality.

This script verifies the quality of element extraction results produced by step2.
For each sample, it takes three images — the original frame (before editing),
the edited frame (after editing), and the extracted element image — and sends
them to a Vision-Language Model (VLM) to assess whether the extraction meets
quality criteria.

For each extracted image, an extraction check is performed using three images
(original frame, edited frame, extracted element) to verify that the third
image correctly extracts the target element from the edited image. A unified
prompt is used that works for all edit types (background change, object
replacement, object addition, etc.) without requiring explicit type
categorisation.

Input:
    --step1_json:  A JSONL file produced by step1 (edit quality verification).
        Each line is a JSON object containing:
            - videoid:              unique identifier for the video pair
            - condition_video_path: path to the before-edit video
            - gt_video_path:        path to the after-edit video
            - condition_frame_path: path to the first frame of the condition video
            - gt_frame_path:        path to the first frame of the gt video
            - instruction:          the edit instruction
            - new_element:          description of the newly introduced element
            - is_edit_success:      "yes" / "no"
            - is_preserve:          "yes" / "no"
        Only entries with is_edit_success == "yes", is_preserve == "yes",
        and a valid new_element are processed.

    --step2_dir:   Directory containing extracted element images from step2,
        named as {videoid}_{prompt_id}.png

Output (--output):
    A JSONL file where each line is a JSON object containing:
        - videoid:              unique identifier for the video pair
        - condition_video_path: path to the before-edit video
        - gt_video_path:        path to the after-edit video
        - condition_frame_path: path to the first frame of the condition video
        - gt_frame_path:        path to the first frame of the gt video
        - new_element:          description of the newly introduced element
        - instruction:          the edit instruction
        - extract_imgs:         list of dicts, each containing:
            - prompt_id:         extraction prompt variant index
            - extracted_path:    path to the extracted element image
            - extraction_result: "yes" / "no" — verification result
        - has_pass:             whether any extracted image passed verification

    This file can be directly used as the --step3_json for step4.py.

Usage:
    python step3.py \\
        --step1_json /path/to/step1_output.jsonl \\
        --step2_dir /path/to/step2_extracted_images/ \\
        --output /path/to/step3_result.jsonl \\
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

EXTRACTION_CHECK_PROMPT = '''You are provided with three images. The second image is an edited version of the first image according to the following instruction. The third image aims to extract the newly introduced or changed element from the second image.

Edit instruction: {instruction}
Target element to extract: {new_element}

You need to make the following judgments regarding the content of the third image:
1. The third image correctly extracts the target element ({new_element}) from the second image, rather than from the first image.
2. The extracted element is exactly the SAME element as the corresponding newly introduced or changed element in the second image.
3. The extracted element is the only main subject in the third image.
4. The extracted element must refer to a complete entity, NOT a partial component or a body part of a subject.
5. If the extracted element is a human character, he or she should appear harmoniously within the third image.
6. Important Note: If the second image differs from the first image only in clothing with the subject remaining the same, the third image must extract only the clothing and MUST NOT extract the person or subject wearing it.

If the third image fully meets the above requirements, directly return "yes" in the "result" field without any extra explanation; otherwise, return "no".

### Output Format:
{{"result":"yes/no"}}
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

def send_extraction_check(url: str, model_name: str,
                          img1_path: str, img2_path: str, img3_path: str,
                          prompt_text: str, max_new_tokens: int, timeout: int = 300):
    """Send a 3-image extraction verification request to the VLM.

    Returns:
        (success, error_msg, result_str) where result_str is "yes" or "no".
    """
    try:
        b64_1 = encode_image(img1_path, max_side=640)
        b64_2 = encode_image(img2_path, max_side=640)
        b64_3 = encode_image(img3_path, max_side=640)
    except Exception as e:
        return False, f"Image encoding error: {e}", None

    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "This is the first image (before editing):"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_1}"}},
                {"type": "text", "text": "This is the second image (after editing):"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_2}"}},
                {"type": "text", "text": "This is the third image (extracted element):"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_3}"}},
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
                result = parsed['result'].lower()
                if result in ('yes', 'no'):
                    return True, "", result
            else:
                print(f"Extraction check attempt {attempt + 1}: HTTP {resp.status_code}")
                if attempt >= 5:
                    return False, f"HTTP {resp.status_code}", None
        except Exception as e:
            print(f"Extraction check attempt {attempt + 1} failed: {e}")
            if attempt >= 5:
                return False, str(e), None

    return False, "Max retries reached", None



# ────────────────────────── data loading ──────────────────────────

def load_input_data(step1_json_path: str, step2_dir: str, num_prompts: int) -> list:
    """Load valid entries from the step1 output JSONL and locate step2 extracted images.

    Keeps only entries where:
        - is_edit_success == "yes" and is_preserve == "yes"
        - new_element is non-empty and not "none"
        - condition_frame_path and gt_frame_path exist on disk
        - at least one extracted image from step2 exists

    Returns:
        A list of dicts, each containing step1 fields plus
        'extracted_images': [(prompt_id, image_path), ...].
    """
    items = []
    with open(step1_json_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Filter by edit quality
            if entry.get('is_edit_success', '').lower() != 'yes':
                continue
            if entry.get('is_preserve', '').lower() != 'yes':
                continue

            new_element = entry.get('new_element', '').strip()
            if not new_element or new_element.lower() == 'none':
                continue

            # Verify frame paths exist
            condition_frame = entry.get('condition_frame_path', '')
            gt_frame = entry.get('gt_frame_path', '')
            if not condition_frame or not os.path.exists(condition_frame):
                continue
            if not gt_frame or not os.path.exists(gt_frame):
                continue

            # Find extracted images from step2
            videoid = entry['videoid']
            extracted = []
            for pid in range(num_prompts):
                img_path = os.path.join(step2_dir, f"{videoid}_{pid}.png")
                if os.path.exists(img_path):
                    extracted.append((pid, img_path))

            if not extracted:
                continue

            items.append({
                'videoid': videoid,
                'condition_video_path': entry.get('condition_video_path', ''),
                'gt_video_path': entry.get('gt_video_path', ''),
                'condition_frame_path': condition_frame,
                'gt_frame_path': gt_frame,
                'new_element': new_element,
                'instruction': entry.get('instruction', ''),
                'extracted_images': extracted,
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
    """Process a single entry: verify all its extracted images.

    Returns:
        (result_dict, has_any_pass): result_dict is None on complete failure.
    """
    videoid = item['videoid']
    condition_frame = item['condition_frame_path']
    gt_frame = item['gt_frame_path']
    new_element = item['new_element']
    instruction = item['instruction']
    extracted_images = item['extracted_images']

    # Build the extraction check prompt
    check_prompt = EXTRACTION_CHECK_PROMPT.format(
        instruction=instruction, new_element=new_element
    )

    results = []
    for pid, img_path in extracted_images:
        # Extraction quality check (3 images)
        ok, err, result = send_extraction_check(
            api_url, model_name, condition_frame, gt_frame, img_path,
            check_prompt, max_new_tokens
        )
        if not ok:
            continue

        results.append({
            'prompt_id': pid,
            'extracted_path': img_path,
            'extraction_result': result,
        })

    if not results:
        return None, False

    # Determine overall pass status
    has_pass = any(r['extraction_result'] == 'yes' for r in results)

    result_dict = {
        'videoid': videoid,
        'condition_video_path': item.get('condition_video_path', ''),
        'gt_video_path': item.get('gt_video_path', ''),
        'condition_frame_path': condition_frame,
        'gt_frame_path': gt_frame,
        'new_element': new_element,
        'instruction': instruction,
        'extract_imgs': results,
        'has_pass': has_pass,
    }

    return result_dict, has_pass


def process_all(all_items, save_path, api_url, model_name, max_new_tokens):
    """Process all pending items and append results to the output JSONL file."""
    done_ids = load_existing_results(save_path)
    todo_items = [item for item in all_items if item['videoid'] not in done_ids]

    print(f'Total: {len(all_items)}, already done: {len(done_ids)}, todo: {len(todo_items)}')

    pass_count = 0
    fail_count = 0

    for k, item in enumerate(tqdm(todo_items)):
        result_dict, has_pass = process_single_item(
            item, api_url, model_name, max_new_tokens
        )

        now_str = datetime.now().strftime("%H:%M:%S")

        if result_dict is not None:
            with open(save_path, 'a', encoding='utf8') as f:
                f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')

            if has_pass:
                pass_count += 1

            # Print per-image results
            for r in result_dict['extract_imgs']:
                print(f"  prompt_id={r['prompt_id']}, "
                      f"extraction={r['extraction_result']}")

            print(f'{now_str} [{k + 1}/{len(todo_items)}] '
                  f'pass={pass_count}, fail={fail_count}')
        else:
            fail_count += 1
            print(f'{now_str} [{k + 1}/{len(todo_items)}] '
                  f'failed (total failures: {fail_count})')

    print(f'\nDone — pass={pass_count}, fail={fail_count}, '
          f'total_processed={len(todo_items)}')


# ────────────────────────── entry point ──────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Step3: Extraction quality verification using VLM")
    p.add_argument("--step1_json", type=str, required=True,
                   help="Path to step1 output JSONL file (edit verification results)")
    p.add_argument("--step2_dir", type=str, required=True,
                   help="Directory containing extracted element images from step2, "
                        "named as {videoid}_{prompt_id}.png")
    p.add_argument("--output", type=str, required=True,
                   help="Path to output JSONL file for saving verification results")
    p.add_argument("--server_ip", type=str, required=True,
                   help="VLM server IP address")
    p.add_argument("--server_port", type=str, default="8080",
                   help="VLM server port (default: 8080)")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-235B-A22B-Instruct",
                   help="Model name for the VLM API")
    p.add_argument("--max_new_tokens", type=int, default=4096,
                   help="Max new tokens for generation (default: 4096)")
    p.add_argument("--num_extraction_prompts", type=int, default=4,
                   help="Number of extraction prompt variants used in step2 (default: 4)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load input data and locate extracted images
    all_items = load_input_data(
        args.step1_json, args.step2_dir, args.num_extraction_prompts
    )
    print(f'Loaded {len(all_items)} valid items with extracted images')

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
