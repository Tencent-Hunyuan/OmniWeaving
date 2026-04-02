"""
Step1: Motion Description Generation

This script takes an input JSONL file containing video metadata (image path and
video path), sends each image-video pair to a VLM API to generate motion
descriptions in both English and Chinese, describing the motion that occurs in
the video relative to the initial frame image.

The output of this script serves as the input to step2.py, which further
corrects the motion descriptions and predicts the underlying intent.

Input (--input_json):
    A JSONL file where each line is a JSON object containing:
        - videoid:  unique identifier for the video
        - video:        path to the video file
    The initial frame image is automatically extracted from the video
    and cached under --frame_cache_dir.

Output (--output):
    A JSONL file where each line is a JSON object with:
        - videoid:  unique identifier
        - image:        path to the initial frame image
        - video:        path to the video file
        - motion_en:     generated English motion description
        - motion_cn:     generated Chinese motion description

    This file can be directly used as the --input_json for step2.py.

Usage:
    python step1.py \\
        --input_json /path/to/input.jsonl \\
        --output /path/to/step1_output.jsonl \\
        --frame_cache_dir /path/to/frame_cache/ \\
        --server_ip vllm_ip \\
        --server_port vllm_port \\
        --model_name "Qwen/Qwen3-VL-235B-A22B-Instruct" \\
        --workers 4
"""

import json
import os
import re
import io
import base64
import argparse
import threading
from datetime import datetime

import numpy as np
import requests
from PIL import Image
from decord import VideoReader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# ────────────────────────── prompts ──────────────────────────

PROMPT_EN = '''
You are a professional video motion analyst. Given an initial image (the first frame) and a video, please carefully observe the video and describe the motion that occurs in the video relative to the initial image.

### Requirements:
1. Focus on what objects or subjects move, how they move, and any significant changes in the scene compared to the initial image.
2. Include relevant details such as direction of movement, interactions between objects, and changes in posture or expression.
3. Directly output the motion description without any extra explanation.
4. Output in English.

### Output format:
{{"Motion_Description":"..."}}
'''

PROMPT_CN = '''
你是一个专业的视频运动分析师。给定一张初始图像（第一帧）和一段视频，请仔细观察视频，描述视频相对于初始图像发生的运动。

### 要求：
1. 关注运动的对象或主体、运动方式以及与初始图像相比场景中的显著变化。
2. 包含相关细节，如运动方向、物体之间的交互、姿态或表情的变化等。
3. 直接输出运动描述，不要有任何多余的解释。
4. 使用中文输出。

### 输出格式：
{{"运动描述":"..."}}
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
    """Extract and parse JSON content from the model's response string.

    Handles fenced code blocks (```json ... ```), raw JSON strings,
    and strips invisible Unicode characters before parsing.
    """
    if not content_str:
        return None

    content_str = content_str.strip()

    # Strip thinking model output if present
    if '</think>' in content_str:
        content_str = content_str.split('</think>')[-1].strip()

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


def extract_first_frame(video_path: str, save_path: str) -> str:
    """Extract the first frame from a video and save it to disk.

    If save_path already exists, skips extraction. Returns save_path.
    """
    if os.path.exists(save_path):
        return save_path

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

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    pil_image.save(save_path)
    return save_path


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


def encode_video(path: str) -> str:
    """Read a video file and return its base64-encoded string."""
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


# ────────────────────────── API helpers ──────────────────────────

def _build_payload(model_name: str, image_b64: str, video_b64: str,
                   prompt_text: str, max_new_tokens: int) -> dict:
    """Build the VLM API request payload with image + video + text."""
    return {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}, "fps": 4.0},
                {"type": "text", "text": prompt_text},
            ]
        }],
        "max_tokens": max_new_tokens,
    }


def _parse_en_response(content_str: str):
    """Parse English response and return the motion description string, or None."""
    parsed = extract_json_from_content(content_str)
    if parsed and parsed.get("Motion_Description"):
        return parsed["Motion_Description"]
    return None


def _parse_cn_response(content_str: str):
    """Parse Chinese response and return the motion description string, or None."""
    parsed = extract_json_from_content(content_str)
    if parsed and parsed.get("运动描述"):
        return parsed["运动描述"]
    return None


def send_description_request(url: str, model_name: str, image_path: str, video_path: str,
                             max_new_tokens: int, idx: int = 0, timeout: int = 300):
    """Send both EN and CN motion description requests for one video.

    Returns (success, error_msg, result_tuple):
        result_tuple = (motion_en, motion_cn, idx)
    """
    # Encode image and video
    try:
        image_b64 = decode_image(image_path)
        video_b64 = encode_video(video_path)
    except Exception as e:
        return False, f"Media encoding error: {e}", None

    payload_en = _build_payload(model_name, image_b64, video_b64, PROMPT_EN, max_new_tokens)
    payload_cn = _build_payload(model_name, image_b64, video_b64, PROMPT_CN, max_new_tokens)

    motion_en = None
    motion_cn = None

    max_retries = 6
    for attempt in range(max_retries):
        try:
            # Request English motion description if not yet obtained
            if not motion_en:
                resp = requests.post(url, json=payload_en, timeout=timeout)
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    motion_en = _parse_en_response(content)

            # Request Chinese motion description if not yet obtained
            if not motion_cn:
                resp = requests.post(url, json=payload_cn, timeout=timeout)
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    motion_cn = _parse_cn_response(content)

            # Return success if both descriptions are obtained
            if motion_en and motion_cn:
                return True, "", (motion_en, motion_cn, idx)

        except Exception as e:
            print(f"  attempt {attempt + 1}/{max_retries} error: {e}")

        if attempt >= max_retries - 1:
            return False, "Max retries reached", None

    return False, "", None


# ────────────────────────── data loading ──────────────────────────

def load_input_data(input_json_path: str, frame_cache_dir: str) -> list:
    """Load valid entries from the input JSONL file.

    Each entry must have: videoid, video.
    The initial frame image is extracted from the video and cached under
    frame_cache_dir. Entries with missing fields or non-existent video
    files are skipped.
    Returns a list of dicts ready for processing.
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

            # Validate required fields
            videoid = entry.get('videoid')
            video_path = entry.get('video', '')

            if not videoid:
                continue
            if not video_path or not os.path.exists(video_path):
                continue

            # Extract the first frame from video as the image
            frame_name = os.path.splitext(os.path.basename(video_path))[0] + '.png'
            image_path = os.path.join(frame_cache_dir, frame_name)
            try:
                extract_first_frame(video_path, image_path)
            except Exception as e:
                print(f"Error extracting first frame for {videoid}: {e}")
                continue

            entry['image'] = image_path
            items.append(entry)

    return items


def load_existing_results(save_path: str) -> set:
    """Load the set of already-processed videoids from the output file to enable resumption."""
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

def process_all(all_items: list, save_path: str, api_url: str, model_name: str,
                max_new_tokens: int, workers: int):
    """Process all pending items, with optional multi-threading.

    Args:
        all_items:      list of input entries to process.
        save_path:      path to output JSONL file (supports append-based resumption).
        api_url:        VLM API endpoint URL.
        model_name:     model name for the VLM API.
        max_new_tokens: max new tokens for generation.
        workers:        number of parallel workers (1 = sequential).
    """
    # Load already-completed entries to support resumption
    done_ids = load_existing_results(save_path)
    todo_items = [item for item in all_items if item['videoid'] not in done_ids]

    print(f'Total: {len(all_items)}, already done: {len(done_ids)}, todo: {len(todo_items)}')

    if not todo_items:
        print('Nothing to process.')
        return

    success_count = 0
    fail_count = 0
    write_lock = threading.Lock()

    def _build_result(md, motion_en, motion_cn):
        """Build the result dict for a single item (compatible with step2 input)."""
        return {
            'videoid': md['videoid'],
            'image': md['image'],
            'video': md['video'],
            'motion_en': motion_en,
            'motion_cn': motion_cn,
        }

    def _handle_result(result, idx):
        """Handle a single API result: write to file and print progress (thread-safe)."""
        nonlocal success_count, fail_count
        now_str = datetime.now().strftime("%H:%M:%S")

        if result[0]:
            motion_en, motion_cn, idd = result[2]
            md = todo_items[idd]
            curres = _build_result(md, motion_en, motion_cn)

            with write_lock:
                with open(save_path, 'a', encoding='utf8') as f:
                    f.write(json.dumps(curres, ensure_ascii=False) + '\n')
                success_count += 1

            print(f'{now_str}: success {success_count}/{len(todo_items)} | '
                  f'en: {motion_en[:80]}... | cn: {motion_cn[:80]}...')
        else:
            with write_lock:
                fail_count += 1
            print(f'{now_str}: fail {fail_count}/{len(todo_items)} | {result[1]}')

    if workers <= 1:
        # Sequential processing
        for kk, md in enumerate(tqdm(todo_items)):
            result = send_description_request(
                api_url, model_name, md['image'], md['video'],
                max_new_tokens, idx=kk,
            )
            _handle_result(result, kk)
    else:
        # Multi-threaded processing
        tasks = [
            (api_url, model_name, md['image'], md['video'],
             max_new_tokens, kk)
            for kk, md in enumerate(todo_items)
        ]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(send_description_request, *task): task[-1]
                for task in tasks
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                result = fut.result()
                _handle_result(result, idx)

    print(f'\nDone — success={success_count}, fail={fail_count}')


# ────────────────────────── entry point ──────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Step1: Motion description generation from video and initial frame")
    p.add_argument("--input_json", type=str, required=True,
                   help="Path to the input JSONL file. Each line is a JSON object "
                        "with videoid, video fields")
    p.add_argument("--output", type=str, required=True,
                   help="Path to output JSONL file for saving results "
                        "(can be used as --input_json for step2.py)")
    p.add_argument("--frame_cache_dir", type=str, required=True,
                   help="Directory to cache extracted first frames from videos")
    p.add_argument("--server_ip", type=str, required=True,
                   help="VLM server IP address")
    p.add_argument("--server_port", type=str, default="8080",
                   help="VLM server port (default: 8080)")
    p.add_argument("--model_name", type=str,
                   default="Qwen/Qwen3-VL-235B-A22B-Instruct",
                   help="Model name for the VLM API")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel workers (1 = sequential, default: 1)")
    p.add_argument("--max_new_tokens", type=int, default=4096,
                   help="Max new tokens for generation (default: 4096)")
    return p.parse_args()


def main():
    args = parse_args()

    # Ensure the frame cache directory exists
    os.makedirs(args.frame_cache_dir, exist_ok=True)

    # Load input data (extracts first frames from videos)
    all_items = load_input_data(args.input_json, args.frame_cache_dir)
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
        workers=args.workers,
    )


if __name__ == '__main__':
    main()
