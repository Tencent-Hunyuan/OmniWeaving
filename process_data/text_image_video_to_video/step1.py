"""
Step1: Edit Quality Verification

This script verifies the quality of image/video editing results by comparing
the first frames of before-edit and after-edit videos using a Vision-Language
Model (VLM). It uses a unified prompt that works for all edit types (background
change, object replacement, object addition, style transfer, etc.) without
requiring explicit type categorisation.

Note: Ideally, different VLM prompts should be designed for different edit
meta-types (e.g., background replacement, object addition, object replacement,
style transfer) to achieve more accurate and targeted verification. 
Here we give a single unified prompt for simplicity and generality.

For each sample, it:
  1. Extracts the first frame from both the condition (before-edit) video and
     the ground-truth (after-edit) video.
  2. Sends the two frames along with the edit instruction to the VLM.
  3. Assesses three aspects:
       - Whether the intended edit was successfully applied.
       - Whether unrelated regions are properly preserved.
       - A short description of the newly introduced or changed element.

Input (--input_json):
    A JSON file containing a list of dicts. Each dict must include:
        - videoid:              unique identifier for the video pair
        - condition_video_path: path to the before-edit video
        - gt_video_path:        path to the after-edit video
        - instruction:          text describing the intended edit

Output (--output):
    A JSONL file where each line is a JSON object containing:
        - videoid:              unique identifier for the video pair
        - index:                original index in the input list
        - condition_video_path: path to the before-edit video
        - gt_video_path:        path to the after-edit video
        - instruction:          the edit instruction
        - is_edit_success:      "yes" / "no" — whether the edit was applied
        - is_preserve:          "yes" / "no" — whether unrelated regions are preserved
        - new_element:          short description of the newly introduced element
        - condition_frame_path: path to the cached first frame of the condition video
                                (present when --frame_cache_dir is specified)
        - gt_frame_path:        path to the cached first frame of the gt video
                                (present when --frame_cache_dir is specified)

    This file can be directly used as the --input_json for step2.py.

Usage:
    python step1.py \\
        --input_json /path/to/data.json \\
        --output /path/to/result.jsonl \\
        --server_ip vllm_ip \\
        --server_port vllm_port \\
        --model_name "Qwen/Qwen3-VL-235B-A22B-Instruct" \\
        --frame_cache_dir /path/to/frame_cache
"""

import json
import os
import re
import io
import base64
import argparse
from datetime import datetime

import numpy as np
import requests
from PIL import Image
from decord import VideoReader
from tqdm import tqdm


# ────────────────────────── prompts ──────────────────────────

EDIT_CHECK_PROMPT = '''You are provided with two images, which are the first frames of two videos before and after a video editing operation. The first image is the first frame of the original video (before editing), and the second image is the first frame of the edited video (after editing). The following instruction describes the intended video editing operation.

Instruction: {instr}

Please directly answer the following three questions without any extra explanation or other words. All judgments about the video editing result below must be made based on the corresponding first frames.
1. Determine if the edit described in the instruction has been successfully applied in the second image. This includes any type of edit such as background replacement, object addition, object replacement, style change, etc. If the edit was successfully applied, return "yes" in the "is_edit_success" field; otherwise, return "no".
2. Determine if the second image preserves the original visual content in regions unrelated to the edit (i.e., areas that should not be affected remain unchanged). Note that very minor variations are permissible. If there are no significant unintended changes, return "yes" in the "is_preserve" field; otherwise, return "no".
3. Provide a short phrase describing the newly introduced or changed element in the second image compared to the first image. For example, if the edit involves replacing or adding an object, describe that object using a short phrase of a few words; if the edit involves changing the background, directly return "the background of the image". This description must allow the new element to be identified even if one can only see the second image. Return this description in the "new_element" field. If no new element can be identified, return "none".

Directly output the answer for each question, without any extra explanation.

### Output Format:
{{"is_edit_success":"yes/no", "is_preserve":"yes/no", "new_element":"..."}}
'''


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


def encode_image(im: Image.Image, max_side=None) -> str:
    """Encode a PIL Image to a base64 string, optionally down-scaling by the longest side."""
    if max_side and max_side > 0:
        w, h = im.size
        scale = max(w, h) / float(max_side)
        if scale > 1:
            im = im.resize((int(round(w / scale)), int(round(h / scale))), Image.BICUBIC)

    buffer = io.BytesIO()
    im.save(buffer, format='JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


def extract_first_frame(video_path: str, save_path: str = None) -> Image.Image:
    """Extract the first frame from a video and optionally cache it to disk.

    Returns:
        A PIL Image in RGB format.
    """
    vr = VideoReader(video_path)
    first_frame = vr.get_batch([0])

    if hasattr(first_frame, 'asnumpy'):
        frame_np = first_frame.asnumpy()
    else:
        frame_np = np.array(first_frame)

    # Remove batch dimension: (1, H, W, C) -> (H, W, C)
    if frame_np.ndim == 4:
        frame_np = frame_np[0]

    if frame_np.dtype != np.uint8:
        frame_np = frame_np.astype(np.uint8)

    pil_image = Image.fromarray(frame_np).convert('RGB')

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        if not os.path.exists(save_path):
            pil_image.save(save_path)

    return pil_image


# ────────────────────────── API helpers ──────────────────────────

def send_edit_check_request(url: str, model_name: str, img1_b64: str, img2_b64: str,
                            prompt_text: str, max_new_tokens: int, timeout: int = 300):
    """Send an edit verification request with two images to the VLM.

    Returns:
        (success, error_msg, parsed_result) where parsed_result is a tuple
        (is_edit_success, is_preserve, new_element) on success.
    """
    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "This is the first image (before editing):"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1_b64}"}},
                {"type": "text", "text": "This is the second image (after editing):"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2_b64}"}},
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
                return True, "", (
                    parsed['is_edit_success'],
                    parsed['is_preserve'],
                    parsed['new_element'],
                )
            else:
                print(f"Attempt {attempt + 1}: HTTP {resp.status_code}")
                if attempt >= 5:
                    return False, f"HTTP {resp.status_code}: {resp.text}", None
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt >= 5:
                return False, str(e), None

    return False, "Max retries reached", None


# ────────────────────────── data loading ──────────────────────────

def load_input_data(input_json_path: str) -> list:
    """Load the data list from the input JSON file.

    The JSON file should be a list of dicts. Required fields per entry:
        videoid, condition_video_path, gt_video_path, instruction.
    Entries with missing fields or non-existent video files are skipped.
    """
    with open(input_json_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    items = []
    for idx, entry in enumerate(data):
        videoid = entry.get('videoid', '')
        condition_video_path = entry.get('condition_video_path', '')
        gt_video_path = entry.get('gt_video_path', '')
        instruction = entry.get('instruction', '')

        if not videoid or not condition_video_path or not gt_video_path or not instruction:
            continue
        if not os.path.exists(condition_video_path) or not os.path.exists(gt_video_path):
            continue

        items.append({
            'videoid': videoid,
            'condition_video_path': condition_video_path,
            'gt_video_path': gt_video_path,
            'instruction': instruction,
            'index': idx,
        })

    return items


def load_existing_results(save_path: str) -> set:
    """Load already-processed videoids from the output file to support resumption."""
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

def _frame_cache_path(video_path: str, cache_dir: str) -> str:
    """Generate a cache file path for the first frame of a video."""
    base = os.path.splitext(os.path.basename(video_path))[0] + '.png'
    return os.path.join(cache_dir, base)


def process_single_item(item, api_url, model_name, max_new_tokens, frame_cache_dir):
    """Process a single video pair: extract frames -> VLM check -> return results.

    Returns:
        (result_dict, success): result_dict is None on failure.
    """
    v1_path = item['condition_video_path']
    v2_path = item['gt_video_path']
    instruction = item['instruction']

    # Determine frame cache paths (None if caching is disabled)
    v1_cache = _frame_cache_path(v1_path, frame_cache_dir) if frame_cache_dir else None
    v2_cache = _frame_cache_path(v2_path, frame_cache_dir) if frame_cache_dir else None

    # Extract first frames from both videos
    try:
        frame1 = extract_first_frame(v1_path, v1_cache)
    except Exception as e:
        print(f"Error extracting frame from condition video: {e}")
        return None, False

    try:
        frame2 = extract_first_frame(v2_path, v2_cache)
    except Exception as e:
        print(f"Error extracting frame from gt video: {e}")
        return None, False

    # Encode frames to base64
    img1_b64 = encode_image(frame1, max_side=640)
    img2_b64 = encode_image(frame2, max_side=640)

    # Send to VLM
    prompt = EDIT_CHECK_PROMPT.format(instr=instruction)
    ok, err, result = send_edit_check_request(
        api_url, model_name, img1_b64, img2_b64, prompt, max_new_tokens
    )

    if not ok:
        return None, False

    is_edit_success, is_preserve, new_element = result
    result_dict = {
        'videoid': item['videoid'],
        'index': item['index'],
        'condition_video_path': v1_path,
        'gt_video_path': v2_path,
        'instruction': instruction,
        'is_edit_success': is_edit_success,
        'is_preserve': is_preserve,
        'new_element': new_element,
    }

    if v1_cache:
        result_dict['condition_frame_path'] = v1_cache
    if v2_cache:
        result_dict['gt_frame_path'] = v2_cache

    return result_dict, True


def process_all(all_items, save_path, api_url, model_name, max_new_tokens, frame_cache_dir):
    """Process all pending items and append results to the output JSONL file."""
    done_ids = load_existing_results(save_path)
    todo_items = [item for item in all_items if item['videoid'] not in done_ids]

    print(f'Total: {len(all_items)}, already done: {len(done_ids)}, todo: {len(todo_items)}')

    pass_count = 0
    fail_count = 0

    for k, item in enumerate(tqdm(todo_items)):
        result_dict, ok = process_single_item(
            item, api_url, model_name, max_new_tokens, frame_cache_dir
        )

        now_str = datetime.now().strftime("%H:%M:%S")

        if ok and result_dict is not None:
            with open(save_path, 'a', encoding='utf8') as f:
                f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')

            is_pass = (result_dict['is_edit_success'].lower() == 'yes'
                       and result_dict['is_preserve'].lower() == 'yes')
            if is_pass:
                pass_count += 1

            print(f'{now_str} [{k + 1}/{len(todo_items)}] '
                  f'edit_success={result_dict["is_edit_success"]}, '
                  f'preserve={result_dict["is_preserve"]}, '
                  f'new_element={result_dict["new_element"]} '
                  f'(pass: {pass_count}/{k + 1})')
        else:
            fail_count += 1
            print(f'{now_str} [{k + 1}/{len(todo_items)}] '
                  f'failed (total failures: {fail_count})')

    print(f'\nDone — pass={pass_count}, fail={fail_count}, '
          f'total_processed={len(todo_items)}')


# ────────────────────────── entry point ──────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Step1: Edit quality verification using VLM")
    p.add_argument("--input_json", type=str, required=True,
                   help="Path to input JSON file. Must be a list of dicts, each with: "
                        "videoid, condition_video_path, gt_video_path, instruction")
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
    p.add_argument("--frame_cache_dir", type=str, default=None,
                   help="Directory to cache extracted first frames as PNG files. "
                        "If not specified, frames are not saved to disk.")
    return p.parse_args()


def main():
    args = parse_args()

    # Load input data
    all_items = load_input_data(args.input_json)
    print(f'Loaded {len(all_items)} valid items from {args.input_json}')

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Ensure the frame cache directory exists (if specified)
    if args.frame_cache_dir:
        os.makedirs(args.frame_cache_dir, exist_ok=True)

    api_url = f'http://{args.server_ip}:{args.server_port}/v1/chat/completions'

    process_all(
        all_items=all_items,
        save_path=args.output,
        api_url=api_url,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        frame_cache_dir=args.frame_cache_dir,
    )


if __name__ == '__main__':
    main()
