"""
Step4: Assemble Valid Results into Arrow File

This script reads the checked results from step3's output JSONL and assembles
all valid entries into a single Apache Arrow file for downstream use.

Filtering criteria for an entry to be included:
  1. All four check fields must be "yes":
     check_motion_en, check_intent_en, check_motion_cn, check_intent_cn
  2. The corrected motion descriptions (new_motion_en, new_motion_cn) and
     intents (intent_en, intent_cn) must be non-empty and not placeholder "..."
  3. The image and video files must exist

Field renaming in the output:
  - intent_en     → prompt_en
  - intent_cn     → prompt_cn
  - new_motion_en → reasoning_trace_en
  - new_motion_cn → reasoning_trace_cn
  - motion_en / motion_cn are NOT stored in the output

Input (--input_json):
    A JSONL file produced by step3.py. Each line is a JSON object containing:
        - videoid:          unique identifier for the video
        - image:            path to the initial frame image
        - video:            path to the video file
        - new_motion_en:    corrected English motion description
        - new_motion_cn:    corrected Chinese motion description
        - intent_en:        predicted English intent
        - intent_cn:        predicted Chinese intent
        - check_motion_en:  "yes" / "no"
        - check_intent_en:  "yes" / "no"
        - check_motion_cn:  "yes" / "no"
        - check_intent_cn:  "yes" / "no"

Output (--output):
    A single Apache Arrow file with the following columns:
        - videoid:             str — unique identifier
        - image:               str — path to the initial frame image
        - video:               str — path to the video file
        - prompt_en:           str — English intent (renamed from intent_en)
        - prompt_cn:           str — Chinese intent (renamed from intent_cn)
        - reasoning_trace_en:  str — corrected English motion description
        - reasoning_trace_cn:  str — corrected Chinese motion description

Usage:
    python step4_make_arrow.py \\
        --input_json /path/to/step3_output.jsonl \\
        --output /path/to/final_output.arrow
"""

import json
import os
import argparse

import pyarrow as pa
import pyarrow.ipc as ipc
from tqdm import tqdm


# ────────────────────────── data loading ──────────────────────────

def load_input_data(input_json_path: str) -> list:
    """Load all entries from the step3 output JSONL file.

    No filtering is applied here; filtering is done during processing.
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
            items.append(entry)
    return items


# ────────────────────────── filtering ──────────────────────────

CHECK_FIELDS = ["check_motion_en", "check_intent_en", "check_motion_cn", "check_intent_cn"]

VALUE_FIELDS = ["new_motion_en", "new_motion_cn", "intent_en", "intent_cn"]


def is_valid_entry(entry: dict) -> bool:
    """Check whether an entry passes all filtering criteria.

    Returns True if:
      - All four check fields are "yes"
      - All value fields are non-empty and not the placeholder "..."
      - The image and video files exist
    """
    # Verify all checks passed
    for field in CHECK_FIELDS:
        if entry.get(field, "no").lower() != "yes":
            return False

    # Verify all value fields are non-empty and not placeholder
    for field in VALUE_FIELDS:
        val = entry.get(field, "")
        if not val or val == "...":
            return False

    # Verify image and video files exist
    image_path = entry.get("image", "")
    video_path = entry.get("video", "")
    if not image_path or not os.path.exists(image_path):
        return False
    if not video_path or not os.path.exists(video_path):
        return False

    return True


# ────────────────────────── main processing ──────────────────────────

def process_entries(entries: list) -> list:
    """Filter entries and transform into rows suitable for Arrow output.

    Returns a list of dicts with renamed fields.
    """
    valid_rows = []

    for entry in tqdm(entries, desc="Filtering entries"):
        if not is_valid_entry(entry):
            continue

        row = {
            "videoid": entry["videoid"],
            "image": entry["image"],
            "video": entry["video"],
            "prompt_en": entry["intent_en"],
            "prompt_cn": entry["intent_cn"],
            "reasoning_trace_en": entry["new_motion_en"],
            "reasoning_trace_cn": entry["new_motion_cn"],
        }
        valid_rows.append(row)

    return valid_rows


def write_arrow(rows: list, output_path: str):
    """Write a list of row dicts to an Apache Arrow file."""
    if not rows:
        print("No valid rows to write.")
        return

    # Define schema
    schema = pa.schema([
        ("videoid", pa.string()),
        ("image", pa.string()),
        ("video", pa.string()),
        ("prompt_en", pa.string()),
        ("prompt_cn", pa.string()),
        ("reasoning_trace_en", pa.string()),
        ("reasoning_trace_cn", pa.string()),
    ])

    # Build arrays
    arrays = [
        pa.array([r["videoid"] for r in rows], type=pa.string()),
        pa.array([r["image"] for r in rows], type=pa.string()),
        pa.array([r["video"] for r in rows], type=pa.string()),
        pa.array([r["prompt_en"] for r in rows], type=pa.string()),
        pa.array([r["prompt_cn"] for r in rows], type=pa.string()),
        pa.array([r["reasoning_trace_en"] for r in rows], type=pa.string()),
        pa.array([r["reasoning_trace_cn"] for r in rows], type=pa.string()),
    ]

    table = pa.table(arrays, schema=schema)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with ipc.new_file(output_path, schema) as writer:
        writer.write_table(table)

    print(f"Saved {len(rows)} valid records to {output_path}")


# ────────────────────────── entry point ──────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Step4: Assemble valid results into Arrow file")
    p.add_argument("--input_json", type=str, required=True,
                   help="Path to the step3 output JSONL file")
    p.add_argument("--output", type=str, required=True,
                   help="Path to the output Arrow file")
    return p.parse_args()


def main():
    args = parse_args()

    # Load input data from step3 output
    all_entries = load_input_data(args.input_json)
    print(f"Loaded {len(all_entries)} entries from {args.input_json}")

    # Filter and transform
    valid_rows = process_entries(all_entries)
    print(f"\nValid entries: {len(valid_rows)} / {len(all_entries)}")

    # Write arrow file
    write_arrow(valid_rows, args.output)


if __name__ == "__main__":
    main()
