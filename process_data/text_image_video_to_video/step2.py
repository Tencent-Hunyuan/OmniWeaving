"""
Step2: New Element Extraction via Image Generation

Note: Ideally, different image generation prompts should be designed for
different edit meta-types (e.g., background replacement, object addition,
object replacement, style transfer) to achieve more accurate and targeted
element extraction. Here we give a general set of prompt
templates for simplicity and generality.

This script takes the output of step1 (edit quality verification results) and
uses a Flux image generation model to extract the newly introduced or changed
element from the edited frame as a standalone subject image.

For each valid entry (where the edit was successful and unrelated content was
preserved), multiple prompt variants are used to generate extraction images.
Some prompts use only the edited frame, while others use both the original and
edited frames as input. The generation is parallelized across multiple GPUs
using multiprocessing.

Input (--input_json):
    A JSONL file produced by step1. Each line is a JSON object containing:
        - videoid:              unique identifier
        - condition_video_path: path to the before-edit video
        - gt_video_path:        path to the after-edit video
        - instruction:          the edit instruction
        - is_edit_success:      "yes" / "no"
        - is_preserve:          "yes" / "no"
        - new_element:          short description of the newly introduced element
        - condition_frame_path: (optional) path to the cached first frame of the condition video
        - gt_frame_path:        (optional) path to the cached first frame of the gt video
    Only entries with is_edit_success == "yes", is_preserve == "yes", and a
    valid (non-empty, non-"none") new_element are processed.

Output (--output_dir):
    A directory where extracted element images are saved, named as:
        {videoid}_{prompt_id}.png
    These images are consumed by step3.py for extraction quality verification.

Usage:
    python step2.py \\
        --input_json /path/to/step1_output.jsonl \\
        --output_dir /path/to/extracted_images/ \\
        --model_path /path/to/FLUX.2-klein-9B/ \\
        --num_gpus 8 \\
        --base_seed 0
"""

import json
import os
import argparse
import traceback
import multiprocessing as mp
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from decord import VideoReader
from diffusers import Flux2KleinPipeline
from tqdm import tqdm


# ────────────────────────── extraction prompts ──────────────────────────

# Each entry is (input_mode, prompt_template).
#   input_mode: "single" = only the edited frame; "pair" = both original and edited frames.
#   {element} is replaced with the new_element description from step1.
EXTRACTION_PROMPTS = [
    ("single",
     "Extract '{element}' from the image as the only subject without any other "
     "subject, centered over a new open background."),
    ("single",
     "Extract '{element}' from the image as the only subject without any other "
     "subject, centered over a new real-world background."),
    ("pair",
     "The second image is an edited version of the first image. Extract the newly "
     "introduced element '{element}' from the second image as the only subject "
     "without any other subject, centered over a new open background."),
    ("pair",
     "The second image is an edited version of the first image. Extract the newly "
     "introduced element '{element}' from the second image as the only subject "
     "without any other subject, centered over a new real-world background."),
]


# ────────────────────────── utilities ──────────────────────────

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


# ────────────────────────── data loading ──────────────────────────

def load_input_data(input_json_path: str, frame_cache_dir: str = None) -> list:
    """Load and filter valid entries from the step1 output JSONL file.

    Keeps only entries where:
        - is_edit_success == "yes"
        - is_preserve == "yes"
        - new_element is non-empty and not "none"

    For each valid entry, resolves frame image paths: uses cached paths from
    step1 if available, otherwise extracts frames from videos into frame_cache_dir.
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

            # Filter by edit quality
            if entry.get('is_edit_success', '').lower() != 'yes':
                continue
            if entry.get('is_preserve', '').lower() != 'yes':
                continue

            new_element = entry.get('new_element', '').strip()
            if not new_element or new_element.lower() == 'none':
                continue

            # Resolve frame image paths
            condition_frame = entry.get('condition_frame_path', '')
            gt_frame = entry.get('gt_frame_path', '')

            # If frame paths are missing or files don't exist, extract from videos
            if (not condition_frame or not os.path.exists(condition_frame)
                    or not gt_frame or not os.path.exists(gt_frame)):
                if not frame_cache_dir:
                    print(f"Skipping {entry.get('videoid', '?')}: "
                          f"no frame paths and no --frame_cache_dir specified")
                    continue

                v1 = entry.get('condition_video_path', '')
                v2 = entry.get('gt_video_path', '')
                if not v1 or not v2 or not os.path.exists(v1) or not os.path.exists(v2):
                    continue

                v1_name = os.path.splitext(os.path.basename(v1))[0] + '.png'
                v2_name = os.path.splitext(os.path.basename(v2))[0] + '.png'
                condition_frame = os.path.join(frame_cache_dir, v1_name)
                gt_frame = os.path.join(frame_cache_dir, v2_name)

                try:
                    extract_first_frame(v1, condition_frame)
                    extract_first_frame(v2, gt_frame)
                except Exception as e:
                    print(f"Error extracting frames for {entry.get('videoid', '?')}: {e}")
                    continue

            if not os.path.exists(condition_frame) or not os.path.exists(gt_frame):
                continue

            items.append({
                'videoid': entry['videoid'],
                'condition_frame_path': condition_frame,
                'gt_frame_path': gt_frame,
                'new_element': new_element,
                'instruction': entry.get('instruction', ''),
            })

    return items


def collect_tasks(all_items: list, output_dir: str, base_seed: int) -> list:
    """Expand each valid entry into individual image generation tasks.

    For each entry, creates one task per extraction prompt variant.
    Tasks whose output images already exist are skipped (supports resumption).
    """
    all_tasks = []

    for entry in all_items:
        videoid = entry['videoid']
        condition_frame = entry['condition_frame_path']
        gt_frame = entry['gt_frame_path']
        new_element = entry['new_element']

        for pid, (input_mode, prompt_template) in enumerate(EXTRACTION_PROMPTS):
            output_path = os.path.join(output_dir, f"{videoid}_{pid}.png")

            if os.path.exists(output_path):
                continue

            prompt = prompt_template.format(element=new_element)

            if input_mode == "single":
                imgs_input = [gt_frame]
            else:  # "pair"
                imgs_input = [condition_frame, gt_frame]

            all_tasks.append({
                'imgs_input': imgs_input,
                'prompt': prompt,
                'output_path': output_path,
                'seed': base_seed,
            })

    return all_tasks


# ────────────────────────── GPU worker functions ──────────────────────────

# Process-level globals for the model (each worker process gets its own copy)
_process_model = None
_process_gpu_id = None


def init_worker(gpu_id: int, model_path: str, torch_dtype):
    """Initialize the worker process: load the Flux model onto the assigned GPU."""
    global _process_model, _process_gpu_id
    _process_gpu_id = gpu_id
    pipe = Flux2KleinPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
    pipe = pipe.to(torch.device(f'cuda:{gpu_id}'))
    _process_model = pipe
    print(f"[GPU {gpu_id}] Model initialized in process {os.getpid()}")


def process_single_task(task_data: dict) -> str:
    """Run a single image generation task and save the result."""
    global _process_model, _process_gpu_id
    gpu_id = _process_gpu_id

    try:
        pipe = _process_model
        device = f"cuda:{gpu_id}"

        imgs_input = task_data['imgs_input']
        prompt = task_data['prompt']
        output_path = task_data['output_path']
        seed = task_data['seed']

        if os.path.exists(output_path):
            return f"[GPU {gpu_id}] Already exists: {os.path.basename(output_path)}"

        cur_img_input = [Image.open(p).convert('RGB') for p in imgs_input]

        image = pipe(
            prompt=prompt,
            image=cur_img_input,
            height=cur_img_input[0].height,
            width=cur_img_input[0].width,
            guidance_scale=1.0,
            num_inference_steps=4,
            generator=torch.Generator(device=device).manual_seed(seed),
        ).images[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        return f"[GPU {gpu_id}] Success: {os.path.basename(output_path)}"

    except Exception as e:
        error_msg = f"[GPU {gpu_id}] Error: {task_data.get('output_path', 'unknown')}: {e}"
        print(error_msg)
        traceback.print_exc()
        return error_msg


def worker_with_init(gpu_id_and_tasks: tuple) -> list:
    """Worker entry point: load model once, then process all assigned tasks."""
    gpu_id, tasks, model_path, torch_dtype = gpu_id_and_tasks
    init_worker(gpu_id, model_path, torch_dtype)

    results = []
    success_count = 0

    for j, task in enumerate(tasks):
        result = process_single_task(task)
        results.append(result)
        if "Success:" in result:
            success_count += 1
        if (j + 1) % 10 == 0 or (j + 1) == len(tasks):
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] [GPU {gpu_id}] {j + 1}/{len(tasks)} tasks, "
                  f"{success_count} successful")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [GPU {gpu_id}] Completed: {success_count}/{len(tasks)} successful")
    return results


# ────────────────────────── entry point ──────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Step2: New element extraction via Flux image generation")
    p.add_argument("--input_json", type=str, required=True,
                   help="Path to the step1 output JSONL file")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save extracted element images")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to the Flux model (e.g., FLUX.2-klein-9B)")
    p.add_argument("--frame_cache_dir", type=str, default=None,
                   help="Directory to cache extracted first frames, used when "
                        "step1 was run without --frame_cache_dir")
    p.add_argument("--num_gpus", type=int, default=8,
                   help="Number of GPUs to use for parallel generation (default: 8)")
    p.add_argument("--base_seed", type=int, default=0,
                   help="Base random seed for generation (default: 0)")
    p.add_argument("--total_shards", type=int, default=1,
                   help="Total number of shards for distributed processing (default: 1)")
    p.add_argument("--shard_id", type=int, default=0,
                   help="Current shard index, 0-based (default: 0)")
    return p.parse_args()


def main():
    args = parse_args()

    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Load and filter input data
    if args.frame_cache_dir:
        os.makedirs(args.frame_cache_dir, exist_ok=True)
    all_items = load_input_data(args.input_json, args.frame_cache_dir)
    print(f'Loaded {len(all_items)} valid items from {args.input_json}')

    # Collect all generation tasks
    os.makedirs(args.output_dir, exist_ok=True)
    all_tasks = collect_tasks(all_items, args.output_dir, args.base_seed)
    print(f'Total {len(all_tasks)} tasks to process')

    if len(all_tasks) == 0:
        print("No tasks to process, exiting.")
        return

    # Apply sharding for distributed processing
    if args.total_shards > 1:
        assert 0 <= args.shard_id < args.total_shards
        shard_start = args.shard_id * len(all_tasks) // args.total_shards
        shard_end = (args.shard_id + 1) * len(all_tasks) // args.total_shards
        all_tasks = all_tasks[shard_start:shard_end]
        print(f'Shard {args.shard_id}/{args.total_shards}: {len(all_tasks)} tasks')

    # Distribute tasks across GPUs round-robin
    num_gpus = args.num_gpus
    gpu_tasks = [[] for _ in range(num_gpus)]
    for idx, task in enumerate(all_tasks):
        gpu_tasks[idx % num_gpus].append(task)

    for gpu_id in range(num_gpus):
        print(f"  GPU {gpu_id}: {len(gpu_tasks[gpu_id])} tasks")

    # Prepare per-GPU arguments
    gpu_args = [
        (gpu_id, gpu_tasks[gpu_id], args.model_path, torch_dtype)
        for gpu_id in range(num_gpus)
    ]

    # Run generation across GPUs
    print(f"Starting multiprocessing with {num_gpus} GPUs...")
    print("Note: Model will be loaded once per GPU process")

    with mp.Pool(processes=num_gpus) as pool:
        gpu_results = list(pool.imap_unordered(worker_with_init, gpu_args))

    # Summarize results
    all_results = [r for gpu_result in gpu_results for r in gpu_result]
    success_count = sum(1 for r in all_results if "Success:" in r)

    print(f"\n{'=' * 60}")
    print(f"Final Summary: {success_count}/{len(all_tasks)} tasks processed successfully")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
