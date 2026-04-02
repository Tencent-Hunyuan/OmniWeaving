"""
Step7: Final Instruction Rewriting with Subject and Background References

This script produces the final rewritten instruction by sending the best
extracted subject images (selected from step5's quality checks) to a VLM,
which replaces symbolic object indicators ([object1], [object2], [object3])
with descriptive noun phrases that reference specific input images.

It combines outputs from multiple previous steps:
  - step2: base data (videoid, objects, rewritten instruction, rewritten_selfcheck)
  - step5: subject extraction quality check (to select the best subject image per object)
  - step6: background extraction quality check (to determine if a usable background exists)

For each valid entry, the script:
  1. Selects the best subject image per object from step5's check results,
     ranked by: check result > subject-background fit > open_background > prompt priority.
  2. Sends the selected subject images along with the rewritten instruction to a VLM,
     which replaces [object1], [object2], [object3] with descriptive noun phrases
     referencing the corresponding images (e.g., "the man in the first image").
  3. Produces two rewritten versions:
     - rewritten_final1: using only subject images (always attempted)
     - rewritten_final2: additionally incorporates a background image reference
       (only attempted if a usable background exists in step6's output)

Note:
    The background image itself is NOT sent to the VLM during rewriting. The
    background-variant prompt simply instructs the VLM to insert a reference to
    a hypothetical background image (e.g., "over the background shown in the
    second/third/fourth image"), which will be used in downstream composition.

Input (--step2_json):
    A JSONL file produced by step2. Each line is a JSON object containing:
        - videoid:              unique identifier
        - objects:              dict with "moving_objects" mapping [objectN] -> description
        - overall:              "yes" from step1's self-check
        - rewritten:            rewritten instruction from step2
        - rewritten_selfcheck:  self-check result for the rewriting

Input (--step5_json):
    A JSONL file produced by step5. Each line contains subject check results:
        - videoid:     unique identifier
        - edited_img:  dict of obj_key -> dict of prompt_id -> [check_results]
        - usable:      "yes" if at least one good subject image per object

Input (--step6_json):
    A JSONL file produced by step6. Each line contains background check results:
        - videoid:     unique identifier
        - background:  dict with "result" and "image" fields
        - bg_usable:   "yes" if the background passed quality checks

Output (--output):
    A JSONL file where each line is a JSON object with:
        - All fields from the step2 entry
        - img_subject:           best subject image info per object (from step5)
        - rewritten_final1:      final instruction using subject images only
        - rewritten_final_flag1: 1 = success, 0 = validation failed, -1 = error
        - img_background:        background info (from step6, if available)
        - rewritten_final2:      final instruction with background reference
        - rewritten_final_flag2: 1 = success, 0 = validation failed, -1 = error

Usage:
    python step7.py \\
        --step2_json /path/to/step2_output.jsonl \\
        --step5_json /path/to/step5_output.jsonl \\
        --step6_json /path/to/step6_output.jsonl \\
        --output /path/to/step7_result.jsonl \\
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


# ────────────────────────── instruction rewriting prompts ──────────────────────────

# For 1 object (subject only) — replaces [object1] with a descriptive noun phrase
INSTRUCTION_PROMPT_1 = '''
You are an English Instruction Rewriting Assistant. Based on the above input image, your task is to replace all symbolic indicators within an input instruction with specific noun phrases while maintaining the original sentence structure.

You are provided with an indicator:
[object1] is the main subject in the input image.

Given an instruction sentence containing this indicator, you must replace all instances of [object1] in the instruction with corresponding descriptive noun phrases (e.g., "the man in the input image").

### Requirement:
1. Do not change any sentence structure. You only need to replace all symbolic indicators (i.e., [object1]) in the sentence with the corresponding descriptive noun phrases of the main subjects in the input image. For example, assuming the main subject in the input image is a man, then [object1] in the original instruction can be replaced with "the man in the image."
2. Do not change the semantic meaning of the original instruction. Strictly Maintain the original sentence structure of the instruction.
3. Ensure the rewritten instruction is fluent. Directly output the rewritten instruction without any extra explanation.

### Examples:
Input Instruction Example: [object1] begins to speak, causing her mouth to open and close and her facial expressions to change dynamically. [object1] smiles, looks around, and nods slightly while talking. Her head also moves subtly. Towards the end, her hands appear at the bottom of the frame, making gestures that accompany her speech.
Output Example: {{"rewritten_instruction": "The woman in the image begins to speak, causing her mouth to open and close and her facial expressions to change dynamically. She smiles, looks around, and nods slightly while talking. Her head also moves subtly. Towards the end, her hands appear at the bottom of the frame, making gestures that accompany her speech."}}

### Input:
Instruction: {ins}

### Output Format
{{"rewritten_instruction":"..."}}
'''

# For 1 object (subject + background reference)
INSTRUCTION_PROMPT_1_BACK = '''
You are an English Instruction Rewriting Assistant. Based on the above input image, your task is to replace all symbolic indicators within an input instruction with specific noun phrases while maintaining the original sentence structure.

You are provided with an indicator:
[object1] is the main subject in the input image.

Given an instruction sentence containing this indicator, you must replace all instances of [object1] in the instruction with corresponding descriptive noun phrases, also specifying it originates from the first image (e.g., "the man in the first image"). Simultaneously, you should also incorporate a brief reference to the background (e.g., "in the background shown in the second image") at an appropriate position within the instruction.

### Requirement:
1. Do not change any sentence structure. You only need to replace all symbolic indicators (i.e., [object1]) in the sentence with the corresponding descriptive noun phrases of the main subjects in the input image, and specify their origin from the first image.
2. Do not change the semantic meaning of the original instruction. Strictly Maintain the original sentence structure of the instruction.
3. For the replaced noun phrases, you need to specify that they come from the first image. For example, assuming the main subject in the image is a man, [object1] in the original instruction can be replaced as "the man in the first image."
4. On this basis, assuming there is a second image defining the background information, please insert a reference to the background at an appropriate position within the rewritten instruction, such as "over the background shown in the second image."
5. Ensure the rewritten instruction is fluent. Directly output the rewritten instruction without any extra explanation.

### Examples:
Input Instruction Example: [object1] begins to speak, causing her mouth to open and close and her facial expressions to change dynamically. [object1] smiles, looks around, and nods slightly while talking. Her head also moves subtly. Towards the end, her hands appear at the bottom of the frame, making gestures that accompany her speech.
Output Example: {{"rewritten_instruction": "The woman in the first image begins to speak over the background shown in the second image., causing her mouth to open and close and her facial expressions to change dynamically. She smiles, looks around, and nods slightly while talking. Her head also moves subtly. Towards the end, her hands appear at the bottom of the frame, making gestures that accompany her speech."}}

### Input:
Instruction: {ins}

### Output Format
{{"rewritten_instruction":"..."}}
'''

# For 2 objects (subject only)
INSTRUCTION_PROMPT_2 = '''
You are an English Instruction Rewriting Assistant. Based on the above two input images, your task is to replace all symbolic indicators within an input instruction with specific noun phrases while maintaining the original sentence structure.

You are provided with two indicators:
[object1] is the main subject in the first image.
[object2] is the main subject in the second image.

Given an instruction sentence containing these indicators, you must replace [object1], and [object2] with descriptive noun phrases that also specify which image the subject comes from (e.g., "the man in the first image").

### Requirement:
1. Do not change any sentence structure. You only need to replace the symbolic indicators [object1], and [object2] with the corresponding descriptive noun phrases of the main subjects in the images and indicate which image they come from (i.e., from the first image, or from the second image).
2. Do not change the semantic meaning of the original instruction. Strictly Maintain the original sentence structure of the instruction.
3. For the replaced noun phrases, you need to specify which image they come from. For example, assuming the main subject in the first image is a man, then [object1] in the original instruction can be replaced as "the man in the first image."
4. Ensure the rewritten instruction is fluent. Directly output the rewritten instruction without any extra explanation.


### Examples:
Input Instruction Example: [object1] and [object2], positioned opposite each other, engage in a synchronized dance routine. [object1] repeatedly raises and lowers her arms, sways her body, and shifts her weight between feet while keeping her back mostly turned. [object2], who is visible face-on, performs identical fluid arm movements, gesturing upwards and outwards, mirroring every motion precisely. Both dancers execute a series of coordinated hand-and-arm motions that flow into each other's actions.
Output Example: {{"rewritten_instruction": "The woman in the first image and the man in the second image, positioned opposite each other, engage in a synchronized dance routine. The woman in the first image repeatedly raises and lowers her arms, sways her body, and shifts her weight between feet while keeping her back mostly turned. The man in the second image, who is visible face-on, performs identical fluid arm movements, gesturing upwards and outwards, mirroring every motion precisely. Both dancers execute a series of coordinated hand-and-arm motions that flow into each other's actions."}}

### Input:
Instruction: {ins}

### Output Format
{{"rewritten_instruction":"..."}}
'''

# For 2 objects (subject + background reference)
INSTRUCTION_PROMPT_2_BACK = '''
You are an English Instruction Rewriting Assistant. Based on the above two input images, your task is to replace all symbolic indicators within an input instruction with specific noun phrases while maintaining the original sentence structure.

You are provided with two indicators:
[object1] is the main subject in the first image.
[object2] is the main subject in the second image.

Given an instruction sentence containing these indicators, you must replace [object1], and [object2] with descriptive noun phrases that also specify which image the subject comes from (e.g., "the man in the first image"). Simultaneously, you should also incorporate a brief reference to the background (e.g., "in the background shown in the third image") at an appropriate position within the instruction.

### Requirement:
1. Do not change any sentence structure. You only need to replace the symbolic indicators [object1], and [object2] with the corresponding descriptive noun phrases of the main subjects in the images and indicate which image they come from (i.e., from the first image, or from the second image).
2. Do not change the semantic meaning of the original instruction. Strictly Maintain the original sentence structure of the instruction.
3. For the replaced noun phrases, you need to specify which image they come from. For example, assuming the main subject in the first image is a man, then [object1] in the original instruction can be replaced as "the man in the first image."
4. On this basis, assuming there is a third image defining the background information, please insert a reference to the background at an appropriate position within the rewritten instruction, such as "over the background shown in the third image."
5. Ensure the rewritten instruction is fluent. Directly output the rewritten instruction without any extra explanation.

### Examples:
Input Instruction Example: [object1] and [object2], positioned opposite each other, engage in a synchronized dance routine. [object1] repeatedly raises and lowers her arms, sways her body, and shifts her weight between feet while keeping her back mostly turned. [object2], who is visible face-on, performs identical fluid arm movements, gesturing upwards and outwards, mirroring every motion precisely. Both dancers execute a series of coordinated hand-and-arm motions that flow into each other's actions.
Output Example: {{"rewritten_instruction": "The woman in the first image and the man in the second image, positioned opposite each other, engage in a synchronized dance routine over the background shown in the third image. The woman in the first image repeatedly raises and lowers her arms, sways her body, and shifts her weight between feet while keeping her back mostly turned. The man in the second image, who is visible face-on, performs identical fluid arm movements, gesturing upwards and outwards, mirroring every motion precisely. Both dancers execute a series of coordinated hand-and-arm motions that flow into each other's actions."}}

### Input:
Instruction: {ins}

### Output Format
{{"rewritten_instruction":"..."}}
'''

# For 3 objects (subject only)
INSTRUCTION_PROMPT_3 = '''
You are an English Instruction Rewriting Assistant. Based on the above three input images, your task is to replace all symbolic indicators within an input instruction with specific noun phrases while maintaining the original sentence structure.

You are provided with three indicators:
[object1] is the main subject in the first image.
[object2] is the main subject in the second image.
[object3] is the main subject in the third image.

Given an instruction sentence containing these indicators, you must replace [object1], [object2], and [object3] with descriptive noun phrases that also specify which image the subject comes from (e.g., "the man in the first image").

### Requirement:
1. Do not change any sentence structure. You only need to replace the symbolic indicators [object1], [object2], and [object3] with the corresponding descriptive noun phrases of the main subjects in the images and indicate which image they come from (i.e., from the first image, or from the second image, or from the third image).
2. Do not change the semantic meaning of the original instruction. Strictly Maintain the original sentence structure of the instruction.
3. For the replaced noun phrases, you need to specify which image they come from. For example, assuming the main subject in the first image is a man, then [object1] in the original instruction can be replaced as "the man in the first image."
4. Ensure the rewritten instruction is fluent. Directly output the rewritten instruction without any extra explanation.


### Examples:
Input Instruction Example: [object1] runs forward with great energy, his face open-mouthed in excitement. Running beside him are [object2] and [object3]. [object2] smiles widely while keeping his fists clenched, and [object3] runs with determination. A crowd of teammates stands behind them, continuously applauding.
Output Example: {{"rewritten_instruction": "The man in the first image runs forward with great energy, his face open-mouthed in excitement. Running beside him are the boy in the second image and the dog in the third image. The boy in the second image smiles widely while keeping his fists clenched, and the dog in the third image runs with determination. A crowd of teammates stands behind them, continuously applauding."}}

### Input:
Instruction: {ins}

### Output Format
{{"rewritten_instruction":"..."}}
'''

# For 3 objects (subject + background reference)
INSTRUCTION_PROMPT_3_BACK = '''
You are an English Instruction Rewriting Assistant. Based on the above three input images, your task is to replace all symbolic indicators within an input instruction with specific noun phrases while maintaining the original sentence structure.

You are provided with three indicators:
[object1] is the main subject in the first image.
[object2] is the main subject in the second image.
[object3] is the main subject in the third image.

Given an instruction sentence containing these indicators, you must replace [object1], [object2], and [object3] with descriptive noun phrases that also specify which image the subject comes from (e.g., "the man in the first image"). Simultaneously, you should also incorporate a brief reference to the background (e.g., "in the background shown in the fourth image") at an appropriate position within the instruction.

### Requirement:
1. Do not change any sentence structure. You only need to replace the symbolic indicators [object1], [object2], and [object3] with the corresponding descriptive noun phrases of the main subjects in the images and indicate which image they come from (i.e., from the first image, or from the second image, or from the third image).
2. Do not change the semantic meaning of the original instruction. Maintain the original sentence structure of the instruction.
3. For the replaced noun phrases, you need to specify which image they come from. For example, assuming the main subject in the first image is a man, then [object1] in the original instruction can be replaced as "the man in the first image."
4. On this basis, assuming there is a fourth image defining the background information, please insert a reference to the background at an appropriate position within the rewritten instruction, such as "over the background shown in the fourth image."
5. Ensure the rewritten instruction is fluent. Directly output the rewritten instruction without any extra explanation.

### Examples:
Input Instruction Example: [object1] runs forward with great energy. Running beside him are [object2] and [object3]. [object2] smiles widely while keeping his fists clenched, and [object3] runs with determination. A crowd of teammates stands behind them, continuously applauding.
Output Example: {{"rewritten_instruction": "The man in the first image runs forward with great energy in the background of the fourth image. Running beside him are the boy in the second image and the dog in the third image. The boy in the second image smiles widely while keeping his fists clenched, and the dog in the third image runs with determination. A crowd of teammates stands behind them, continuously applauding."}}

### Input:
Instruction: {ins}

### Output Format
{{"rewritten_instruction":"..."}}
'''

# Select appropriate prompts based on number of objects
PROMPTS_BY_COUNT = {
    1: (INSTRUCTION_PROMPT_1, INSTRUCTION_PROMPT_1_BACK),
    2: (INSTRUCTION_PROMPT_2, INSTRUCTION_PROMPT_2_BACK),
    3: (INSTRUCTION_PROMPT_3, INSTRUCTION_PROMPT_3_BACK),
}

VALID_OBJECT_KEYS = {"[object1]", "[object2]", "[object3]"}

# Priority weights for selecting the best subject image from step5 results.
# Higher values are preferred as tiebreakers when result/fit/open_background are equal.
# Keys are step3 prompt IDs (0-3). Adjust these based on your step3 prompt variants.
PROMPT_PRIORITY = {0: 3, 1: 2, 2: 0, 3: 1}

QUALITY_WEIGHT = {'yes': 1, 'no': -1}

# Ordinal words used to validate that the rewritten instruction references
# the expected number of images
ORDINAL_WORDS = ['first', 'second', 'third', 'fourth']


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


def validate_rewrite(instruction: str, expected_image_count: int) -> bool:
    """Check that the rewritten instruction contains ordinal references to the expected images.

    For single-object prompts without background (count=1), always returns True
    since the prompt uses "the image" instead of ordinals.
    For multi-image prompts, verifies all expected ordinals are present.
    """
    if expected_image_count <= 1:
        return True

    instruction_lower = instruction.lower()
    for i in range(expected_image_count):
        if i >= len(ORDINAL_WORDS):
            break
        if ORDINAL_WORDS[i] not in instruction_lower:
            return False
    return True


# ────────────────────────── data loading ──────────────────────────

def load_step2_data(path: str) -> list:
    """
    Load valid entries from step2 output JSONL.

    Filters to entries with:
    - overall == "yes" (passed step1's check)
    - rewritten_selfcheck.result == "yes" (passed step2's self-check)
    - Non-empty rewritten instruction
    - Non-empty, valid moving_objects
    """
    items = []
    with open(path, 'r', encoding='utf8') as f:
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

            rewritten = entry.get("rewritten")
            if not rewritten:
                continue

            try:
                objects = entry['objects']['moving_objects']
            except (KeyError, TypeError):
                continue
            if not objects:
                continue
            if not all(k in VALID_OBJECT_KEYS for k in objects.keys()):
                continue

            items.append(entry)

    return items


def _select_best_subject(edited_img: dict) -> dict:
    """
    Select the best subject image for each object from step5 check results.

    For each object, candidates are ranked by:
      1. Check result (yes > no)
      2. Subject-background fit (yes > no)
      3. Open background (yes > no)
      4. Prompt priority (higher is better)

    Returns dict: obj_key -> best check result dict.
    """
    choice = {}
    for obj_key in sorted(edited_img.keys()):
        candidates = []
        for pid_str, results_list in edited_img[obj_key].items():
            pid = int(pid_str) if isinstance(pid_str, str) else pid_str
            for result in results_list:
                result_copy = dict(result)
                result_copy['prompt_priority'] = PROMPT_PRIORITY.get(pid, 0)
                candidates.append(result_copy)

        if not candidates:
            continue

        candidates.sort(
            key=lambda x: (
                QUALITY_WEIGHT.get(x.get('result', 'no').lower(), -1),
                QUALITY_WEIGHT.get(x.get('fit', 'no').lower(), -1),
                QUALITY_WEIGHT.get(x.get('open_background', 'no').lower(), -1),
                x.get('prompt_priority', 0),
            ),
            reverse=True,
        )

        choice[obj_key] = candidates[0]

    return choice


def load_step5_data(path: str) -> dict:
    """
    Load step5 results and select the best subject image per object per videoid.

    Only entries with usable == "yes" are included.

    Returns dict: videoid -> {obj_key: best_check_result_dict}
    """
    subject_map = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("usable", "no").lower() != "yes":
                continue

            videoid = entry.get("videoid")
            edited_img = entry.get("edited_img", {})
            if not videoid or not edited_img:
                continue

            choice = _select_best_subject(edited_img)
            if choice:
                subject_map[videoid] = choice

    return subject_map


def load_step6_data(path: str) -> dict:
    """
    Load step6 results and get usable background info per videoid.

    Only entries with bg_usable == "yes" are included.

    Returns dict: videoid -> background check result dict.
    """
    background_map = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("bg_usable", "no").lower() != "yes":
                continue

            videoid = entry.get("videoid")
            background = entry.get("background", {})
            if not videoid:
                continue

            background_map[videoid] = background

    return background_map


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


# ────────────────────────── API helpers ──────────────────────────

def _build_multi_image_payload(model_name: str, image_b64_list: list,
                               prompt_text: str, max_new_tokens: int) -> dict:
    """Build the VLM API request payload with multiple labeled images.

    Each image is labeled with its ordinal ("This is the first image:" etc.).
    """
    content = []
    for i, img_b64 in enumerate(image_b64_list):
        ordinal = ORDINAL_WORDS[i] if i < len(ORDINAL_WORDS) else f"image {i + 1}"
        content.append({"type": "text", "text": f"This is the {ordinal} image:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })
    content.append({"type": "text", "text": prompt_text})

    return {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_new_tokens,
    }


def send_rewrite_request(url: str, model_name: str, image_b64_list: list,
                         prompt_text: str, max_new_tokens: int,
                         expected_image_count: int, timeout: int = 30):
    """Send a final instruction rewriting request.

    Returns (flag, rewritten_instruction):
        flag:  1 = success, 0 = validation failed, -1 = error
    """
    payload = _build_multi_image_payload(model_name, image_b64_list,
                                         prompt_text, max_new_tokens)
    rewritten = None

    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                # Handle thinking models that wrap output in <think> tags
                if '</think>' in content:
                    content = content.split('</think>')[-1].strip()
                parsed = json.loads(content)
                rewritten = parsed['rewritten_instruction']
                if validate_rewrite(rewritten, expected_image_count):
                    return 1, rewritten
            else:
                print(f"HTTP {resp.status_code}: {resp.text}")
                if attempt >= 2:
                    return (-1 if rewritten is None else 0), rewritten
        except Exception as e:
            print(e)
            if attempt >= 2:
                return (-1 if rewritten is None else 0), rewritten

    # Exhausted retries without a valid result
    if rewritten is not None:
        return 0, rewritten
    return -1, None


# ────────────────────────── main processing ──────────────────────────

def process_single_item(entry: dict, subject_info: dict, background_info,
                        api_url: str, model_name: str, max_new_tokens: int):
    """
    Process a single entry: rewrite instruction with subject (and optional background) references.

    Args:
        entry: step2 output entry.
        subject_info: best subject image per object (from step5), dict: obj_key -> check_result.
        background_info: background check result from step6 (dict or None).
        api_url: VLM API endpoint URL.
        model_name: model name for the API.
        max_new_tokens: max tokens for generation.

    Returns the entry dict enriched with rewriting results.
    """
    objects = entry['objects']['moving_objects']
    rewritten = entry['rewritten']
    num_objects = len(objects)

    if num_objects not in PROMPTS_BY_COUNT:
        print(f'  Unsupported number of objects: {num_objects}')
        entry['rewritten_final_flag1'] = -1
        entry['rewritten_final1'] = None
        return entry

    prompt_subject, prompt_back = PROMPTS_BY_COUNT[num_objects]

    # Encode subject images in sorted object key order
    obj_keys_sorted = sorted(objects.keys())
    image_b64_list = []
    img_load_ok = True

    for obj_key in obj_keys_sorted:
        img_path = subject_info.get(obj_key, {}).get('image', '')
        if not img_path or not os.path.exists(img_path):
            img_load_ok = False
            break
        try:
            image_b64_list.append(decode_image(img_path, 640))
        except Exception as e:
            print(f'  Error encoding image {img_path}: {e}')
            img_load_ok = False
            break

    if not img_load_ok:
        print('  Failed to load subject images')
        entry['rewritten_final_flag1'] = -1
        entry['rewritten_final1'] = None
        return entry

    entry['img_subject'] = subject_info

    # Rewrite 1: subject images only
    formatted_prompt1 = prompt_subject.format(ins=rewritten)
    flag1, rewritten1 = send_rewrite_request(
        api_url, model_name, image_b64_list, formatted_prompt1,
        max_new_tokens, expected_image_count=num_objects,
    )
    entry['rewritten_final_flag1'] = flag1
    entry['rewritten_final1'] = rewritten1
    print(f'  rewritten_final1 (flag={flag1}): {rewritten1}')

    # Rewrite 2: subject images + background reference (if background available)
    if background_info is not None:
        formatted_prompt2 = prompt_back.format(ins=rewritten)
        expected_count_with_back = num_objects + 1
        flag2, rewritten2 = send_rewrite_request(
            api_url, model_name, image_b64_list, formatted_prompt2,
            max_new_tokens, expected_image_count=expected_count_with_back,
        )
        entry['img_background'] = background_info
        entry['rewritten_final_flag2'] = flag2
        entry['rewritten_final2'] = rewritten2
        print(f'  rewritten_final2 (flag={flag2}): {rewritten2}')
    else:
        print('  No usable background — skipping rewrite with background')

    return entry


def process_all(all_items: list, subject_map: dict, background_map: dict,
                save_path: str, api_url: str, model_name: str, max_new_tokens: int):
    """Process all pending items."""
    done_ids = load_existing_results(save_path)
    todo_items = [item for item in all_items if item['videoid'] not in done_ids]

    print(f'Total: {len(all_items)}, already done: {len(done_ids)}, todo: {len(todo_items)}')

    success1_count = 0
    success2_count = 0
    fail_count = 0
    skip_count = 0

    for k, entry in enumerate(tqdm(todo_items)):
        videoid = entry['videoid']
        now_str = datetime.now().strftime("%H:%M:%S")
        print(f'\n{now_str} [{k}/{len(todo_items)}] Processing: {videoid}')
        print(f'  objects: {json.dumps(entry["objects"]["moving_objects"])}')

        # Check that we have subject images from step5
        if videoid not in subject_map:
            print('  Skipping — no usable subject images from step5')
            skip_count += 1
            continue

        subject_info = subject_map[videoid]
        background_info = background_map.get(videoid)

        result = process_single_item(
            entry, subject_info, background_info,
            api_url, model_name, max_new_tokens,
        )

        flag1 = result.get('rewritten_final_flag1', -1)

        if flag1 != -1:
            with open(save_path, 'a', encoding='utf8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

            if flag1 == 1:
                success1_count += 1
            else:
                fail_count += 1

            flag2 = result.get('rewritten_final_flag2', -1)
            if flag2 == 1:
                success2_count += 1
        else:
            fail_count += 1

        total = success1_count + fail_count
        now_str = datetime.now().strftime("%H:%M:%S")
        print(f'{now_str} [{k}/{len(todo_items)}] '
              f'(total={total}, ok1={success1_count}, ok2={success2_count}, '
              f'fail={fail_count}, skip={skip_count})')

    print(f'\nDone — ok1={success1_count}, ok2={success2_count}, '
          f'fail={fail_count}, skipped={skip_count}')


# ────────────────────────── entry point ──────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Step7: Final instruction rewriting with subject and background references")
    p.add_argument("--step2_json", type=str, required=True,
                   help="Path to the step2 output JSONL file (base data with "
                        "videoid, objects, rewritten instruction, rewritten_selfcheck)")
    p.add_argument("--step5_json", type=str, required=True,
                   help="Path to the step5 output JSONL file (subject quality check results "
                        "with edited_img and usable fields)")
    p.add_argument("--step6_json", type=str, required=True,
                   help="Path to the step6 output JSONL file (background quality check results "
                        "with background and bg_usable fields)")
    p.add_argument("--output", type=str, required=True,
                   help="Path to output JSONL file for saving final rewriting results")
    p.add_argument("--server_ip", type=str, required=True, help="VLM server IP address")
    p.add_argument("--server_port", type=str, default="8080",
                   help="VLM server port (default: 8080)")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-235B-A22B-Instruct",
                   help="Model name for the VLM API")
    p.add_argument("--max_new_tokens", type=int, default=4096,
                   help="Max new tokens for generation")
    return p.parse_args()


def main():
    args = parse_args()

    # Load data from all upstream steps
    all_items = load_step2_data(args.step2_json)
    print(f'Loaded {len(all_items)} valid items from step2: {args.step2_json}')

    subject_map = load_step5_data(args.step5_json)
    print(f'Loaded {len(subject_map)} entries with usable subjects from step5: {args.step5_json}')

    background_map = load_step6_data(args.step6_json)
    print(f'Loaded {len(background_map)} entries with usable backgrounds from step6: {args.step6_json}')

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    api_url = f'http://{args.server_ip}:{args.server_port}/v1/chat/completions'

    process_all(
        all_items=all_items,
        subject_map=subject_map,
        background_map=background_map,
        save_path=args.output,
        api_url=api_url,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == '__main__':
    main()
