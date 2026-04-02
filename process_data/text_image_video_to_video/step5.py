"""
Step5: Final Consistency Verification

Note: Ideally, different VLM prompts should be designed for different edit
meta-types (e.g., background replacement, object addition, object replacement,
style transfer) to achieve more targeted final verification.
Here we give a single unified prompt for simplicity and generality.

This script performs a final verification to ensure the full pipeline is
consistent: the rewritten instruction (from step4), the extracted element image
(from step2), and the edited video frame all align correctly.

For each sample, three images are sent to a VLM:
  - First image:  the original frame (before editing)
  - Second image: the extracted element image (the target element to apply)
  - Third image:  the edited frame (after editing)
along with the rewritten instruction from step4.

Two checks are performed:
  1. Whether the rewritten instruction aligns with the extracted element image
     (i.e., the second image actually contains the element described by the
     instruction).
  2. Whether the edited frame (third image) correctly applies the element
     from the second image to the original frame, following the instruction.

A unified prompt is used that works for all edit types (background change,
object replacement, object addition, etc.) without requiring explicit type
categorisation.

Input (--step4_json):
    A JSONL file produced by step4. Each line is a JSON object containing:
        - videoid:               unique identifier for the video pair
        - condition_video_path:  path to the before-edit video
        - gt_video_path:         path to the after-edit video
        - condition_frame_path:  path to the first frame of the condition video
        - gt_frame_path:         path to the first frame of the gt video
        - best_extracted_path:   path to the best extracted element image
        - new_element:           description of the newly introduced element
        - instruction:           the original edit instruction
        - rewritten:             rewritten instruction from step4
        - rewritten_ok:          "yes" / "no" — whether the rewrite is valid
    Only entries with rewritten_ok == "yes" are processed.

Output (--output):
    A JSONL file where each line is a JSON object containing:
        - videoid:               unique identifier for the video pair
        - condition_video_path:  path to the before-edit video
        - gt_video_path:         path to the after-edit video
        - condition_frame_path:  path to the first frame of the condition video
        - gt_frame_path:         path to the first frame of the gt video
        - best_extracted_path:   path to the best extracted element image
        - new_element:           description of the newly introduced element
        - instruction:           the original edit instruction
        - rewritten:             rewritten instruction from step4
        - check1:                "yes" / "no" — instruction aligns with extracted element
        - check2:                "yes" / "no" — edited frame correctly applies the element

    This file can be directly used as the --step5_json for step6_make_arrow.py.

Usage:
    python step5.py \\
        --step4_json /path/to/step4_output.jsonl \\
        --output /path/to/step5_result.jsonl \\
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

FINAL_CHECK_PROMPT = '''You are provided with three images and a video editing instruction. The first image is the initial frame of the original video. The second image provides the target element (e.g., a new background scene, a new object, or a modified subject) to be applied in the editing. The third image is the initial frame of the edited video. The instruction describes how the element from the second image should be applied to the original video.

Instruction: {instr}

You need to make the following judgments based on the three images and the instruction:
1. Assess whether the instruction aligns with the second image. For example, if the instruction describes adding or replacing with a specific element, verify whether the second image actually contains that target element. If the instruction does not describe a specific element in detail, assume that the instruction aligns with the second image. If the instruction aligns with the second image, return "yes" in the "check1" field without any additional explanation; otherwise, return "no".

2. Determine whether the third image correctly applies the element from the second image to the first image in accordance with the instruction. For instance, if the instruction requires replacing the background, check whether the background in the third image matches the scene shown in the second image; if the instruction requires adding or replacing an object, check whether the object in the third image is the same object as the one in the second image. If the third image correctly applies the element from the second image following the instruction, return "yes" in the "check2" field; otherwise, return "no". Note: if the element in the third image is not the same element as the one in the second image, return "no" in the "check2" field.

### Output Format:
{{"check1":"yes/no", "check2":"yes/no"}}
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

def send_final_check(url: str, model_name: str,
                     img1_path: str, img2_path: str, img3_path: str,
                     prompt_text: str, max_new_tokens: int, timeout: int = 300):
    """Send a 3-image final consistency check request to the VLM.

    Images:
        img1 = original frame (before editing)
        img2 = extracted element image
        img3 = edited frame (after editing)

    Returns:
        (success, error_msg, (check1, check2)) where check1/check2 are "yes"/"no".
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
                {"type": "text", "text": "This is the first image (original frame before editing):"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_1}"}},
                {"type": "text", "text": "This is the second image (extracted target element):"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_2}"}},
                {"type": "text", "text": "This is the third image (edited frame after editing):"},
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
                c1 = parsed['check1'].lower()
                c2 = parsed['check2'].lower()
                if c1 in ('yes', 'no') and c2 in ('yes', 'no'):
                    return True, "", (c1, c2)
            else:
                print(f"Final check attempt {attempt + 1}: HTTP {resp.status_code}")
                if attempt >= 5:
                    return False, f"HTTP {resp.status_code}", None
        except Exception as e:
            print(f"Final check attempt {attempt + 1} failed: {e}")
            if attempt >= 5:
                return False, str(e), None

    return False, "Max retries reached", None


# ────────────────────────── data loading ──────────────────────────

def load_input_data(step4_json_path: str) -> list:
    """Load valid entries from the step4 output JSONL file.

    Keeps only entries where:
        - rewritten_ok == "yes"
        - rewritten is non-empty
        - condition_frame_path, gt_frame_path, best_extracted_path all exist on disk
    """
    items = []
    with open(step4_json_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Filter: rewritten must be valid
            if entry.get('rewritten_ok', '').lower() != 'yes':
                continue

            rewritten = entry.get('rewritten', '').strip()
            if not rewritten or rewritten == '...':
                continue

            # Verify all image paths exist
            condition_frame = entry.get('condition_frame_path', '')
            gt_frame = entry.get('gt_frame_path', '')
            best_extracted = entry.get('best_extracted_path', '')

            if not all(p and os.path.exists(p) for p in
                       [condition_frame, gt_frame, best_extracted]):
                continue

            items.append({
                'videoid': entry['videoid'],
                'condition_video_path': entry.get('condition_video_path', ''),
                'gt_video_path': entry.get('gt_video_path', ''),
                'condition_frame_path': condition_frame,
                'gt_frame_path': gt_frame,
                'best_extracted_path': best_extracted,
                'new_element': entry.get('new_element', ''),
                'instruction': entry.get('instruction', ''),
                'rewritten': rewritten,
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
    """Process a single entry: run the final consistency check.

    Returns:
        (result_dict, flag):
            flag:  1 = both checks passed,
                   0 = at least one check failed,
                  -1 = API error
    """
    condition_frame = item['condition_frame_path']
    gt_frame = item['gt_frame_path']
    best_extracted = item['best_extracted_path']
    rewritten = item['rewritten']

    prompt = FINAL_CHECK_PROMPT.format(instr=rewritten)

    # Image order: original frame, extracted element, edited frame
    ok, err, result = send_final_check(
        api_url, model_name,
        condition_frame, best_extracted, gt_frame,
        prompt, max_new_tokens
    )

    if not ok:
        return None, -1

    check1, check2 = result
    both_pass = (check1 == 'yes' and check2 == 'yes')

    result_dict = {
        'videoid': item['videoid'],
        'condition_video_path': item.get('condition_video_path', ''),
        'gt_video_path': item.get('gt_video_path', ''),
        'condition_frame_path': condition_frame,
        'gt_frame_path': gt_frame,
        'best_extracted_path': best_extracted,
        'new_element': item['new_element'],
        'instruction': item['instruction'],
        'rewritten': rewritten,
        'check1': check1,
        'check2': check2,
    }

    return result_dict, 1 if both_pass else 0


def process_all(all_items, save_path, api_url, model_name, max_new_tokens):
    """Process all pending items and append results to the output JSONL file."""
    done_ids = load_existing_results(save_path)
    todo_items = [item for item in all_items if item['videoid'] not in done_ids]

    print(f'Total: {len(all_items)}, already done: {len(done_ids)}, todo: {len(todo_items)}')

    pass_count = 0
    partial_count = 0
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
                pass_count += 1
            else:
                partial_count += 1

            print(f'{now_str} [{k + 1}/{len(todo_items)}] '
                  f'check1={result_dict["check1"]}, check2={result_dict["check2"]} '
                  f'(pass={pass_count}, partial={partial_count}, fail={fail_count})')
        else:
            fail_count += 1
            print(f'{now_str} [{k + 1}/{len(todo_items)}] '
                  f'failed (total failures: {fail_count})')

    print(f'\nDone — pass={pass_count}, partial={partial_count}, '
          f'fail={fail_count}, total_processed={len(todo_items)}')


# ────────────────────────── entry point ──────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Step5: Final consistency verification using VLM")
    p.add_argument("--step4_json", type=str, required=True,
                   help="Path to step4 output JSONL file (rewritten instructions)")
    p.add_argument("--output", type=str, required=True,
                   help="Path to output JSONL file for saving final check results")
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
    all_items = load_input_data(args.step4_json)
    print(f'Loaded {len(all_items)} valid items from {args.step4_json}')

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
