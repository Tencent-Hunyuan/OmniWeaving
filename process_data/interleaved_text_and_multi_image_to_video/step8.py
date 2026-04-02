"""
Step8: Final Instruction Self-Check

This script validates the rewritten instructions produced by step7 by sending
the subject images (and optionally background image) along with the original
frame to a VLM. The VLM determines whether the original frame could serve as
a valid first frame for a video clip described by the rewritten instruction.

Two types of checks are performed per entry:
  1. Check 1 (subjects only): Validates rewritten_final1 by sending subject
     images and the rewritten instruction to the VLM, then presenting the
     original frame as the "target image" for evaluation.
  2. Check 2 (subjects + background): Validates rewritten_final2 by additionally
     including the background image along with subject images before presenting
     the original frame.

The VLM is asked to judge whether the target image correctly extracts and
combines the subjects (and optionally background) from the input images as
described in the instruction, making it a qualified first frame for the video.

Input (--input_json):
    A JSONL file produced by step7. Each line is a JSON object containing:
        - videoid:               unique identifier
        - image_name:            path to the original frame image
        - objects:               dict with "moving_objects" mapping [objectN] -> desc
        - overall:               "yes" from step1's self-check
        - rewritten_selfcheck:   self-check result dict with "result" field
        - img_subject:           dict of obj_key -> {image: path, ...} (best subject per object)
        - rewritten_final1:      rewritten instruction (subjects only)
        - rewritten_final_flag1: 1 = success from step7
        - img_background:        dict with {image: path, result: ...} (optional, from step6)
        - rewritten_final2:      rewritten instruction (with background reference)
        - rewritten_final_flag2: 1 = success from step7

Output (--output):
    A JSONL file where each line is a JSON object with all original fields plus:
        - check_1:          "yes"/"no" — whether rewritten_final1 passed VLM check
        - check_reason_1:   reason from VLM (if provided)
        - check_2:          "yes"/"no" — whether rewritten_final2 passed VLM check
        - check_reason_2:   reason from VLM (if provided)

Usage:
    python step8.py \\
        --input_json /path/to/step7_output.jsonl \\
        --output /path/to/step8_result.jsonl \\
        --server_ip vllm_ip \\
        --server_port vllm_port \\
        --model_name "Qwen/Qwen3-VL-235B-A22B-Instruct"
"""

import json
import os
import io
import base64
import argparse
from datetime import datetime

import requests
from PIL import Image
from tqdm import tqdm


# ────────────────────────── check prompts ──────────────────────────

# Multi-image check prompt — Part 1 (displayed before the target image)
PROMPT_CHECK_MULTI_PART1 = '''
You are a visual assistant. Based on the above {num} images, I will first provide you with an instruction describing how to combine the subject or background information from these images to construct a new scene of video clip.

Instruction: {ins}

Based on the above {num} images and the instruction, I will then provide a new image as the first frame of the target video clip. You need to determine whether this new image is a qualified first frame.

The new image is:
'''

# Multi-image check prompt — Part 2 (displayed after the target image)
PROMPT_CHECK_MULTI_PART2 = '''
### Requirements:
1. Determine whether the newly given image correctly extract and combine the subject and background information from the previously provided images as specified in the instructions, acting as a qualified first frame for the target video clip. If it is a qualified first frame, directly return "yes" in the "result" field, otherwise return "no". Do NOT output any extra explanations.
2. Ignore all descriptions of motion, position, pose, facial expression, or temporal changes within the instructions during your determination. You only need to judge whether the first frame correctly extracts and combines the subjects from the previously provided images.
3. It is allowed to including additional new objects in the new image, which are not present in either of the original images.
4. To judge whether the newly given image conduct a correct subject extraction, you only need to judge whether the extracted object is the same physical objects as the corresponding one in the original images. Any changes in scenario, position, pose, orientation, facial expression, and perspective are entirely allowed for the subjects in the new image compared to the subjects in the original images.

### Output Format
{{"result":"yes/no"}}
'''

# Single-image check prompt — Part 1 (displayed before the target image)
PROMPT_CHECK_SINGLE_PART1 = '''
You are a visual assistant. Based on the above image, I will first provide you with an instruction describing how to leverage the subject from the images to construct a new scene of video clip.

Instruction: {ins}

Based on the above image and the instruction, I will then provide a new image as the first frame of the target video clip. You need to determine whether this new image is a qualified first frame.

The new image is:
'''

# Single-image check prompt — Part 2 (displayed after the target image)
PROMPT_CHECK_SINGLE_PART2 = '''
### Requirements:
1. Determine whether the newly given image correctly extract the subject from the previously provided image to construct a qualified first frame for the target video clip, as specified in the instruction. If it is a qualified first frame, directly return "yes" in the "result" field, otherwise return "no". Do NOT output any extra explanations.
2. Ignore all descriptions of motion, position, pose, facial expression, or temporal changes within the instructions during your determination. You only need to judge whether the first frame correctly extracts the subject from the previously provided image.
3. Including additional new objects in the new image is allowed.
4. To judge whether the newly given image conduct a correct subject extraction, you only need to judge whether the extracted object is the same physical objects as the corresponding one in the original image. Any changes in scenario, position, pose, orientation, facial expression, and perspective are entirely allowed for the subject in the new image compared to the subject in the original image.

### Output Format
{{"result":"yes/no"}}
'''

# Ordinal labels for labeling images in multi-image payloads
ORDINAL_LABELS = ['first', 'second', 'third', 'fourth']

VALID_OBJECT_KEYS = {"[object1]", "[object2]", "[object3]"}


# ────────────────────────── utilities ──────────────────────────

def decode_image(path: str, max_side=None) -> str:
    """Read an image file and return its base64-encoded string, optionally rescaling."""
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

def _build_check_payload(model_name: str, input_b64_list: list,
                         target_b64: str, prompt_part1: str,
                         prompt_part2: str, max_new_tokens: int) -> dict:
    """Build the VLM API request payload for instruction self-check.

    Payload structure:
      [labeled input images] + [prompt_part1] + [target image] + [prompt_part2]

    For a single input image, no ordinal label is added.
    For multiple input images, each is labeled (e.g., "This is the first image:").
    """
    content = []

    # Add labeled input images (subject images, optionally followed by background)
    for i, img_b64 in enumerate(input_b64_list):
        if len(input_b64_list) > 1:
            ordinal = ORDINAL_LABELS[i] if i < len(ORDINAL_LABELS) else f"image {i + 1}"
            content.append({"type": "text", "text": f"This is the {ordinal} image:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })

    # Add prompt part 1 (instruction context)
    content.append({"type": "text", "text": prompt_part1})

    # Add target image (the original frame to be evaluated)
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{target_b64}"},
    })

    # Add prompt part 2 (requirements and output format)
    content.append({"type": "text", "text": prompt_part2})

    return {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_new_tokens,
    }


def send_check_request(url: str, model_name: str, input_b64_list: list,
                       target_b64: str, prompt_part1: str, prompt_part2: str,
                       max_new_tokens: int, timeout: int = 30):
    """Send a self-check request to the VLM.

    Returns (success, result, reason):
        success: True if a valid response was received.
        result:  "yes" or "no" (empty string on failure).
        reason:  explanation from the VLM (empty string if not provided).
    """
    payload = _build_check_payload(model_name, input_b64_list, target_b64,
                                   prompt_part1, prompt_part2, max_new_tokens)

    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                # Handle thinking models that wrap output in <think> tags
                if '</think>' in content:
                    content = content.split('</think>')[-1].strip()
                parsed = json.loads(content)
                result_val = parsed.get('result', '').lower()
                if result_val in ('yes', 'no'):
                    return True, result_val, parsed.get('reason', '')
            else:
                print(f"HTTP {resp.status_code}: {resp.text}")
                if attempt >= 2:
                    return False, '', ''
        except Exception as e:
            print(e)
            if attempt >= 2:
                return False, '', ''

    return False, '', ''


# ────────────────────────── data loading ──────────────────────────

def load_input_data(input_json_path: str) -> list:
    """
    Load valid entries from the step7 output JSONL file.

    Filters to entries with:
    - overall == "yes" (passed step1's check)
    - rewritten_selfcheck.result == "yes" (passed step2's self-check)
    - At least one successful rewrite (rewritten_final_flag1 == 1 or rewritten_final_flag2 == 1)
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

            if entry.get("overall", "").lower() != "yes":
                continue

            selfcheck = entry.get("rewritten_selfcheck", {})
            if selfcheck.get("result", "no").lower() != "yes":
                continue

            flag1 = entry.get("rewritten_final_flag1", 0)
            flag2 = entry.get("rewritten_final_flag2", 0)
            if flag1 != 1 and flag2 != 1:
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

def _load_subject_b64_list(entry: dict) -> list:
    """Load and base64-encode all subject images from the entry in sorted key order.

    Returns a list of base64 strings, or None if any image fails to load.
    """
    img_subject = entry.get("img_subject", {})
    all_possible_keys = ["[object1]", "[object2]", "[object3]"]
    key_num = len(img_subject)
    obj_keys = all_possible_keys[:key_num]

    b64_list = []
    for obj_key in obj_keys:
        img_path = img_subject.get(obj_key, {}).get('image', '')
        if not img_path or not os.path.exists(img_path):
            return None
        try:
            b64_list.append(decode_image(img_path, 640))
        except Exception as e:
            print(f'  Error encoding subject image {img_path}: {e}')
            return None

    return b64_list


def _load_background_b64(entry: dict):
    """Load and base64-encode the background image from the entry.

    Returns the base64 string, or None if unavailable.
    """
    bg_info = entry.get("img_background")
    if not bg_info:
        return None

    bg_path = bg_info.get("image", "")
    if not bg_path or not os.path.exists(bg_path):
        return None
    try:
        return decode_image(bg_path, 640)
    except Exception as e:
        print(f'  Error encoding background image {bg_path}: {e}')
        return None


# Flag value meanings:
#   -0.5 = not attempted (rewritten_final_flagN != 1)
#     -1 = API error during check
#     -2 = resource not loadable (e.g. missing background image)
#      0 = VLM returned "no"
#      1 = VLM returned "yes"

def process_single_item(entry: dict, api_url: str, model_name: str,
                        max_new_tokens: int):
    """
    Process a single entry: check rewritten_final1 and rewritten_final2 using VLM.

    Returns the entry dict enriched with check results, or None on critical failure.
    """
    videoid = entry['videoid']
    image_name = entry.get('image_name', '')

    # Load original frame image (the "target" for VLM evaluation)
    try:
        target_b64 = decode_image(image_name, 640)
    except Exception as e:
        print(f'  Error loading original frame {image_name}: {e}')
        return None

    # Load subject images
    subject_b64_list = _load_subject_b64_list(entry)
    if subject_b64_list is None:
        print(f'  Failed to load subject images for {videoid}')
        return None

    key_num = len(subject_b64_list)

    # --- Check 1: subjects only ---
    flag1 = -0.5
    check1, reason1 = '', ''

    if entry.get("rewritten_final_flag1", 0) == 1:
        instr = entry.get("rewritten_final1", "")
        if not instr:
            flag1 = -1
        else:
            if key_num > 1:
                p1 = PROMPT_CHECK_MULTI_PART1.format(num=key_num, ins=instr)
                p2 = PROMPT_CHECK_MULTI_PART2
            else:
                p1 = PROMPT_CHECK_SINGLE_PART1.format(ins=instr)
                p2 = PROMPT_CHECK_SINGLE_PART2

            success, check1, reason1 = send_check_request(
                api_url, model_name, subject_b64_list, target_b64,
                p1, p2, max_new_tokens
            )
            if success:
                flag1 = 1 if check1 == 'yes' else 0
            else:
                flag1 = -1

    # --- Check 2: subjects + background ---
    flag2 = -0.5
    check2, reason2 = '', ''

    if entry.get("rewritten_final_flag2", 0) == 1:
        instr = entry.get("rewritten_final2", "")
        if not instr:
            flag2 = -1
        else:
            bg_b64 = _load_background_b64(entry)
            if bg_b64 is None:
                flag2 = -2
            else:
                imgs_with_bg = subject_b64_list + [bg_b64]
                p1 = PROMPT_CHECK_MULTI_PART1.format(num=key_num + 1, ins=instr)
                p2 = PROMPT_CHECK_MULTI_PART2

                success, check2, reason2 = send_check_request(
                    api_url, model_name, imgs_with_bg, target_b64,
                    p1, p2, max_new_tokens
                )
                if success:
                    flag2 = 1 if check2 == 'yes' else 0
                else:
                    flag2 = -1

    # Skip entries where both checks failed or either had an API error
    if (flag1 < 0 and flag2 < 0) or flag1 == -1 or flag2 == -1:
        return None

    # Enrich entry with check results
    if flag1 >= 0:
        entry['check_1'] = check1
        entry['check_reason_1'] = reason1
    if flag2 >= 0:
        entry['check_2'] = check2
        entry['check_reason_2'] = reason2

    return entry


def process_all(all_items: list, save_path: str, api_url: str,
                model_name: str, max_new_tokens: int):
    """Process all pending items."""
    done_ids = load_existing_results(save_path)
    todo_items = [item for item in all_items if item['videoid'] not in done_ids]

    print(f'Total: {len(all_items)}, already done: {len(done_ids)}, todo: {len(todo_items)}')

    # Counters: c = passed, w = failed, e = error/skipped
    c1, w1, e1 = 0, 0, 0
    c2, w2, e2 = 0, 0, 0

    for k, entry in enumerate(tqdm(todo_items)):
        videoid = entry['videoid']

        now_str = datetime.now().strftime("%H:%M:%S")
        print(f'\n{now_str} [{k}/{len(todo_items)}] Processing: {videoid}')

        result = process_single_item(entry, api_url, model_name, max_new_tokens)

        if result is None:
            e1 += 1
            e2 += 1
            now_str = datetime.now().strftime("%H:%M:%S")
            print(f'{now_str} [{k}/{len(todo_items)}] '
                  f'[({c1}+{w1}+{e1}), ({c2}+{w2}+{e2})]')
            continue

        # Save result
        with open(save_path, 'a', encoding='utf8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

        # Update counters for check 1
        ch1 = result.get('check_1')
        if ch1 is not None:
            if ch1 == 'yes':
                c1 += 1
            else:
                w1 += 1
        else:
            e1 += 1

        # Update counters for check 2
        ch2 = result.get('check_2')
        if ch2 is not None:
            if ch2 == 'yes':
                c2 += 1
            else:
                w2 += 1
        else:
            e2 += 1

        now_str = datetime.now().strftime("%H:%M:%S")
        print(f'{now_str} [{k}/{len(todo_items)}] '
              f'[({c1}+{w1}+{e1}), ({c2}+{w2}+{e2})]')
        print(f'  check_1: {result.get("check_1", "N/A")}, '
              f'check_2: {result.get("check_2", "N/A")}')

    print(f'\nDone — check1: yes={c1}, no={w1}, error={e1} | '
          f'check2: yes={c2}, no={w2}, error={e2}')


# ────────────────────────── entry point ──────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Step8: Final instruction self-check via VLM")
    p.add_argument("--input_json", type=str, required=True,
                   help="Path to the step7 output JSONL file containing rewritten "
                        "instructions, subject/background images, and rewriting flags")
    p.add_argument("--output", type=str, required=True,
                   help="Path to output JSONL file for saving check results")
    p.add_argument("--server_ip", type=str, required=True,
                   help="VLM server IP address")
    p.add_argument("--server_port", type=str, default="8080",
                   help="VLM server port (default: 8080)")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-235B-A22B-Instruct",
                   help="Model name for the VLM API")
    p.add_argument("--max_new_tokens", type=int, default=4096,
                   help="Max new tokens for generation")
    return p.parse_args()


def main():
    args = parse_args()

    # Load input data from step7 output
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
    )


if __name__ == '__main__':
    main()
