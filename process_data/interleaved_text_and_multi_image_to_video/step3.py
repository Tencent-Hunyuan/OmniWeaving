"""
Step3: Subject Extraction via Image Generation

This script takes the output of step2 (entries with identified moving objects and
rewritten instructions) and uses a Flux image generation model to extract each
identified object from the original frame as a standalone subject image.

For each valid entry and each moving object, multiple prompt variants are used
to generate extraction images. The generation is parallelized across multiple GPUs
using multiprocessing.

Note:
    As described in the paper, using SAM3 for segmentation prior to extraction can
    yield etter results, especially when there are multiple objects
    in the scene. However, for simpler scenarios (e.g., a single dominant subject),
    the SAM3 segmentation step can be skipped without noticeable quality loss.
    This script provides the simplified pipeline that skips SAM3 and directly uses
    Flux2 for subject image extraction. 

Input (--input_json):
    A JSONL file produced by step2. Each line is a JSON object containing:
        - videoid:     unique identifier for the video
        - image_name:  path to the original frame image
        - objects:     dict with "moving_objects" mapping [objectN] -> description
        - overall:     "yes" from step1's self-check
        - rewritten:   rewritten instruction from step2
    Only entries with overall == "yes", valid rewritten instruction, and
    non-empty moving_objects are processed.

Output (--output_dir):
    A directory where extracted subject images are saved, one per object per
    prompt variant, named as: {videoid}_{object_key}_{prompt_id}.png

Usage:
    python step3.py \\
        --input_json /path/to/step2_output.jsonl \\
        --output_dir /path/to/extracted_images/ \\
        --model_path /path/to/FLUX.2-klein-9B/ \\
        --num_gpus 8 \\
        --base_seed 42
"""

import argparse
import json
import os
import traceback
import multiprocessing as mp
from datetime import datetime

import torch
from PIL import Image
from diffusers import Flux2KleinPipeline
from tqdm import tqdm


# ────────────────────────── extraction prompts ──────────────────────────

# Each prompt is a template with {text} placeholder for the object description.
# These prompts instruct the model to extract a specific object as the sole subject.
EXTRACTION_PROMPTS = [
    "Extract {text} from the image as the only subject, centered over a new open background.",
    "Extract {text} from the image as the only subject without any other subject, centered over a real-world background.",
    "Extract {text} from the image as the only subject without any other subject over a new background.",
    "Extract {text} from the image as the only subject without any other subject, centered over a new background.",
]

VALID_OBJECT_KEYS = {"[object1]", "[object2]", "[object3]"}


# ────────────────────────── data loading ──────────────────────────

def load_input_data(input_json_path: str) -> list:
    """
    Load valid entries from the step2 output JSONL file.

    Only entries with overall == "yes", a non-empty rewritten instruction,
    and valid moving_objects are kept.
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

            image_path = entry.get('image_name', '')
            if not image_path or not os.path.exists(image_path):
                continue

            # Validate moving_objects
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


def collect_tasks(all_items: list, output_dir: str, base_seed: int) -> list:
    """
    Expand each input entry into individual image generation tasks.

    For each entry and each moving object, creates one task per prompt variant.
    Tasks whose output images already exist are skipped (supports resumption).
    """
    all_tasks = []

    for entry in all_items:
        videoid = entry['videoid']
        image_path = entry['image_name']
        objects = entry['objects']['moving_objects']

        for obj_key in sorted(objects.keys()):
            obj_desc = objects[obj_key]

            for pid, prompt_template in enumerate(EXTRACTION_PROMPTS):
                output_path = os.path.join(
                    output_dir, f"{videoid}_{obj_key}_{pid}.png"
                )

                if os.path.exists(output_path):
                    continue

                all_tasks.append({
                    'image_path': image_path,
                    'prompt': prompt_template.format(text=obj_desc),
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
    device = torch.device(f'cuda:{gpu_id}')
    pipe = pipe.to(device)
    _process_model = pipe
    print(f"[GPU {gpu_id}] Model initialized in process {os.getpid()}")


def process_single_task(task_data: dict) -> str:
    """Run a single image generation task and save the result."""
    global _process_model, _process_gpu_id
    gpu_id = _process_gpu_id

    try:
        pipe = _process_model
        device = f"cuda:{gpu_id}"

        image_path = task_data['image_path']
        prompt = task_data['prompt']
        output_path = task_data['output_path']
        seed = task_data['seed']

        if os.path.exists(output_path):
            return f"[GPU {gpu_id}] Already exists: {os.path.basename(output_path)}"

        # Load and resize the source image
        source_img = Image.open(image_path).convert('RGB')
        source_img.thumbnail((640, 640))

        # Generate the extracted subject image
        with torch.inference_mode():
            result_img = pipe(
                prompt=prompt,
                image=[source_img],
                height=source_img.height,
                width=source_img.width,
                guidance_scale=1.0,
                num_inference_steps=4,
                generator=torch.Generator(device=device).manual_seed(seed),
            ).images[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_img.save(output_path)

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
    p = argparse.ArgumentParser(description="Step3: Subject extraction via Flux image generation")
    p.add_argument("--input_json", type=str, required=True,
                   help="Path to the step2 output JSONL file. Each line is a JSON object "
                        "with videoid, image_name, objects (including moving_objects), "
                        "overall, and rewritten fields")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save extracted subject images")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to the Flux model (e.g., FLUX.2-klein-9B)")
    p.add_argument("--num_gpus", type=int, default=8,
                   help="Number of GPUs to use for parallel generation")
    p.add_argument("--base_seed", type=int, default=42,
                   help="Base random seed for generation")
    p.add_argument("--num_inference_steps", type=int, default=4,
                   help="Number of inference steps for the Flux model")
    p.add_argument("--guidance_scale", type=float, default=1.0,
                   help="Guidance scale for the Flux model")
    return p.parse_args()


def main():
    args = parse_args()

    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Load input data
    all_items = load_input_data(args.input_json)
    print(f'Loaded {len(all_items)} valid items from {args.input_json}')

    # Collect all generation tasks
    os.makedirs(args.output_dir, exist_ok=True)
    all_tasks = collect_tasks(all_items, args.output_dir, args.base_seed)
    print(f'Total {len(all_tasks)} tasks to process using {args.num_gpus} GPUs')

    if len(all_tasks) == 0:
        print("No tasks to process, exiting.")
        return

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
        gpu_results = []
        with tqdm(total=len(all_tasks), desc="Overall Progress", unit="task") as pbar:
            for gpu_result in pool.imap_unordered(worker_with_init, gpu_args):
                gpu_results.append(gpu_result)
                pbar.update(len(gpu_result))

    # Summarize results
    all_results = [r for gpu_result in gpu_results for r in gpu_result]
    success_count = sum(1 for r in all_results if "Success:" in r)

    print(f"\n{'=' * 60}")
    print(f"Final Summary: {success_count}/{len(all_tasks)} tasks processed successfully")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
