"""
Step2: Motion Description Correction and Intent Prediction

This script takes an input JSONL file containing video metadata (image path,
video path, and motion descriptions in both English and Chinese), sends them to
a VLM API to:
  1. Check and correct the motion descriptions.
  2. Predict the underlying intent behind the motion.

Input (--input_json):
    A JSONL file produced by step1.py, where each line is a JSON object containing:
        - videoid:      unique identifier for the video
        - image:        path to the initial frame image
        - video:        path to the video file
        - motion_en:     English motion description
        - motion_cn:     Chinese motion description

Output (--output):
    A JSONL file where each line is a JSON object with:
        - videoid:         unique identifier
        - image:           path to the initial frame image
        - video:           path to the video file
        - motion_en:       original English motion description
        - motion_cn:       original Chinese motion description
        - new_motion_en:   corrected English motion description
        - new_motion_cn:   corrected Chinese motion description
        - intent_en:       predicted English intent
        - intent_cn:       predicted Chinese intent

    This file can be directly used as the --input_json for step3.py.

Usage:
    python step2.py \\
        --input_json /path/to/input.jsonl \\
        --output /path/to/output.jsonl \\
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

import requests
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# ────────────────────────── prompts ──────────────────────────

PROMPT_CN = '''
你是一个专业的视频意图推测器，给定一张初始图像和一段视频，请结合场景信息与人物表情行为，用一句话隐式地描述视频为何相对于初始图像产生这样的运动。
在生成意图之前，这里是描述视频相对于初始图像的运动描述。您需要检查其正确性并进行适当的修改。

运动描述：{prompt}

步骤一：检查并修正运动描述
要求：
1. 判断给定的运动描述是否正确地描述了视频。如果描述不正确，请修改运动描述，使其正确描述视频相对于初始图像的运动。如果描述正确，直接输出原始运动描述。
2. 如果动作描述中包含英文，请将英文翻译成中文，确保表达流畅并保持原始句式结构。
3. 不要输出任何额外的解释。不要改变原始句式结构。如果原始运动描述已经正确，直接返回它。

步骤二：生成视频运动意图
要求：
1. 直接输出视频运动意图，用一句简短的话概括，务必简洁，不得有任何多余解释，使用中文表达。
2. 不要重复我提供的运动描述，需客观补充一些隐含信息，说明发生了什么。
3. 运动描述是你输出的意图描述的因果结果，即视频中发生的运动是对你所输出意图的正确执行。

输出格式：
{{"运动描述":"...", "意图":"..."}}
'''

PROMPT_EN = '''
You are a professional video intent predictor. Given an initial image and a video, please combine scene context, facial expressions, and character actions, to implicitly describe in one sentence why the video generated such motion relative to the initial image.

Before generating the intent, here is the motion description which describes the motion of the video relative to the initial image. You need to check its correctness and make appropriate modifications.

Motion Description: {prompt}

### Step1: Checking and correcting the motion description
Requirements:
1. Determine if the given motion description correctly describes the motion of the video relative to the initial image. If it does not, modify the motion description **without changing the sentence structure** so that it correctly describes the motion of the video relative to the initial image. Otherwise, directly output the original motion description.
2. If the motion description contains Chinese, translate the Chinese into English, ensuring fluency and maintaining the original sentence structure.
3. Do not output any extra explanation. Do not change the original sentence structure. If the original motion description is already correct, return it directly.

### Step2: Generating the intent of the video motion
Requirements:
1. Directly output the intent of the video motion, summarizing it in a brief sentence, based on the corrected motion description. It must be concise, with no extra explanation.
2. Do not repeat the motion description. You must objectively supplement some implicit information, explaining what happening.
3. The motion description causally follows from the intent you state, which means that the motion occurred in the video is the correct execution of your output intent.

### Output format:
{{"Motion_Description":"...", "Intent":"..."}}
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
    """Parse English response and return (motion_description, intent) or None."""
    parsed = extract_json_from_content(content_str)
    if parsed and parsed.get("Motion_Description") and parsed.get("Intent"):
        return parsed["Motion_Description"], parsed["Intent"]
    return None


def _parse_cn_response(content_str: str):
    """Parse Chinese response and return (motion_description, intent) or None."""
    parsed = extract_json_from_content(content_str)
    if parsed and parsed.get("运动描述") and parsed.get("意图"):
        return parsed["运动描述"], parsed["意图"]
    return None


def send_intent_request(url: str, model_name: str, image_path: str, video_path: str,
                        prompt_en: str, prompt_cn: str, max_new_tokens: int,
                        idx: int = 0, timeout: int = 300):
    """Send both EN and CN intent requests.

    Returns (success, error_msg, result_tuple):
        result_tuple = (new_motion_en, intent_en, new_motion_cn, intent_cn, idx)
    """
    # Encode image and video
    try:
        image_b64 = decode_image(image_path)
        video_b64 = encode_video(video_path)
    except Exception as e:
        return False, f"Media encoding error: {e}", None

    payload_en = _build_payload(model_name, image_b64, video_b64, prompt_en, max_new_tokens)
    payload_cn = _build_payload(model_name, image_b64, video_b64, prompt_cn, max_new_tokens)

    new_motion_en, intent_en = None, None
    new_motion_cn, intent_cn = None, None

    max_retries = 6
    for attempt in range(max_retries):
        try:
            # Request English result if not yet obtained
            if not (new_motion_en and intent_en):
                resp = requests.post(url, json=payload_en, timeout=timeout)
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    result = _parse_en_response(content)
                    if result:
                        new_motion_en, intent_en = result

            # Request Chinese result if not yet obtained
            if not (new_motion_cn and intent_cn):
                resp = requests.post(url, json=payload_cn, timeout=timeout)
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    result = _parse_cn_response(content)
                    if result:
                        new_motion_cn, intent_cn = result

            # Return success if both results are obtained
            if (new_motion_en and intent_en) and (new_motion_cn and intent_cn):
                return True, "", (new_motion_en, intent_en, new_motion_cn, intent_cn, idx)

        except Exception as e:
            print(f"  attempt {attempt + 1}/{max_retries} error: {e}")

        if attempt >= max_retries - 1:
            return False, "Max retries reached", None

    return False, "", None


# ────────────────────────── data loading ──────────────────────────

def load_input_data(input_json_path: str) -> list:
    """Load valid entries from the input JSONL file.

    Each entry must have: videoid, image, video, motion_en, motion_cn.
    Entries with missing fields or non-existent files are skipped.
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
            image_path = entry.get('image', '')
            video_path = entry.get('video', '')
            motion_en = entry.get('motion_en', '')
            motion_cn = entry.get('motion_cn', '')

            if not videoid:
                continue
            if not image_path or not os.path.exists(image_path):
                continue
            if not video_path or not os.path.exists(video_path):
                continue
            if not motion_en or not motion_cn:
                continue

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

    def _build_result(md, new_motion_en, intent_en, new_motion_cn, intent_cn):
        """Build the result dict for a single item."""
        return {
            'videoid': md['videoid'],
            'image': md['image'],
            'video': md['video'],
            'motion_en': md['motion_en'],
            'motion_cn': md['motion_cn'],
            'new_motion_en': new_motion_en,
            'new_motion_cn': new_motion_cn,
            'intent_en': intent_en,
            'intent_cn': intent_cn,
        }

    def _handle_result(result, idx):
        """Handle a single API result: write to file and print progress (thread-safe)."""
        nonlocal success_count, fail_count
        now_str = datetime.now().strftime("%H:%M:%S")

        if result[0]:
            new_motion_en, intent_en, new_motion_cn, intent_cn, idd = result[2]
            md = todo_items[idd]
            curres = _build_result(md, new_motion_en, intent_en, new_motion_cn, intent_cn)

            with write_lock:
                with open(save_path, 'a', encoding='utf8') as f:
                    f.write(json.dumps(curres, ensure_ascii=False) + '\n')
                success_count += 1

            print(f'{now_str}: success {success_count}/{len(todo_items)} | '
                  f'intent_en: {intent_en} | intent_cn: {intent_cn}')
        else:
            with write_lock:
                fail_count += 1
            print(f'{now_str}: fail {fail_count}/{len(todo_items)} | {result[1]}')

    if workers <= 1:
        # Sequential processing
        for kk, md in enumerate(tqdm(todo_items)):
            result = send_intent_request(
                api_url, model_name, md['image'], md['video'],
                PROMPT_EN.format(prompt=md['motion_en']),
                PROMPT_CN.format(prompt=md['motion_cn']),
                max_new_tokens, idx=kk,
            )
            _handle_result(result, kk)
    else:
        # Multi-threaded processing
        tasks = [
            (api_url, model_name, md['image'], md['video'],
             PROMPT_EN.format(prompt=md['motion_en']),
             PROMPT_CN.format(prompt=md['motion_cn']),
             max_new_tokens, kk)
            for kk, md in enumerate(todo_items)
        ]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(send_intent_request, *task): task[-1]
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
        description="Step2: Motion description correction and intent prediction")
    p.add_argument("--input_json", type=str, required=True,
                   help="Path to the input JSONL file. Each line is a JSON object "
                        "with videoid, image, video, motion_en, motion_cn fields")
    p.add_argument("--output", type=str, required=True,
                   help="Path to output JSONL file for saving results")
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
        workers=args.workers,
    )


if __name__ == '__main__':
    main()
