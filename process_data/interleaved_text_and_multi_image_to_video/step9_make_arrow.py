"""
Step9: Assemble Valid Results into Arrow File

This script reads the final checked results from step8's output JSONL and
assembles all valid entries into a single Apache Arrow file for downstream
training / inference use.

Filtering criteria for an entry to be included:
  1. overall == "yes" (step1 check passed)
  2. rewritten_final_flag1 > 0 (step7 successfully rewrote with subjects only)
  3. check_1 == "yes" (step8 validated the subjects-only rewrite)
  4. rewritten_final1 is non-empty
  5. All subject images in img_subject have result == "yes"
     (in --strict_filter mode, also requires fit == "yes")
  6. All subject image files exist and can be opened by PIL

If these checks pass, the entry is included with:
  - subject_img:    list of subject image paths (sorted by object key)
  - prompt1:        rewritten instruction (subjects only)
  - background_img: background image path (if check_2 passed and background is valid)
  - prompt2:        rewritten instruction with background (if background is valid)
  - num_objects:    number of subject objects (1, 2, or 3)

Input (--input_json):
    A JSONL file produced by step8. Each line is a JSON object containing
    all fields carried forward from the pipeline plus check_1 and check_2 results.

Output (--output):
    A single Apache Arrow file containing one row per valid entry with the
    following columns:
        - videoid:        str   — unique identifier
        - image_name:     str   — path to the original frame image
        - video_path:     str   — path to the original video (if available)
        - prompt:         str   — original caption / prompt
        - subject_img:    str   — JSON-encoded list of subject image paths
        - prompt1:        str   — rewritten instruction (subjects only)
        - background_img: str   — background image path (or empty string)
        - prompt2:        str   — rewritten instruction with background (or empty string)
        - num_objects:    int   — number of subject objects

Usage:
    python step9_make_arrow.py \\
        --input_json /path/to/step8_output.jsonl \\
        --output /path/to/final_output.arrow

    # Strict filtering (also requires fit == "yes" for each subject):
    python step9_make_arrow.py \\
        --input_json /path/to/step8_output.jsonl \\
        --output /path/to/final_output_strict.arrow \\
        --strict_filter
"""

import json
import os
import argparse

import pyarrow as pa
import pyarrow.ipc as ipc
from PIL import Image
from tqdm import tqdm


VALID_OBJECT_KEYS = ["[object1]", "[object2]", "[object3]"]


# ────────────────────────── data loading ──────────────────────────

def load_input_data(input_json_path: str) -> list:
    """
    Load all entries from the step8 output JSONL file.

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

def _validate_subject_images(entry: dict, strict_filter: bool) -> list:
    """
    Validate all subject images in the entry.

    Checks:
      - Each subject has result == "yes"
      - In strict mode, each subject also has fit == "yes"
      - Each image file exists and can be opened by PIL

    Returns a list of valid image paths (sorted by object key), or None on failure.
    """
    img_subject = entry.get("img_subject", {})
    all_keys = sorted(img_subject.keys())

    if not all_keys:
        return None

    img_paths = []
    for key in all_keys:
        subject_info = img_subject[key]

        # Check result field
        if subject_info.get("result", "no").lower() != "yes":
            return None

        # In strict mode, also check fit field
        if strict_filter:
            if subject_info.get("fit", "no").lower() != "yes":
                return None

        # Verify image file
        img_path = subject_info.get("image", "")
        if not img_path or not os.path.exists(img_path):
            return None
        try:
            Image.open(img_path).verify()
            img_paths.append(img_path)
        except Exception:
            return None

    return img_paths


def _validate_background(entry: dict) -> tuple:
    """
    Check if a valid background image and prompt2 are available.

    Returns (background_img_path, prompt2) or (None, None) if unavailable.
    """
    if entry.get("rewritten_final_flag2", 0) <= 0:
        return None, None

    if entry.get("check_2", "no").lower() != "yes":
        return None, None

    prompt2 = entry.get("rewritten_final2", "")
    if not prompt2:
        return None, None

    bg_info = entry.get("img_background")
    if not bg_info or not isinstance(bg_info, dict):
        return None, None

    if bg_info.get("result", "no").lower() != "yes":
        return None, None

    bg_path = bg_info.get("image", "")
    if not bg_path or not os.path.exists(bg_path):
        return None, None

    try:
        Image.open(bg_path).verify()
    except Exception:
        return None, None

    return bg_path, prompt2


def is_valid_entry(entry: dict) -> bool:
    """Check the basic validity criteria for an entry (before image checks)."""
    if entry.get("overall", "no").lower() != "yes":
        return False
    if entry.get("rewritten_final_flag1", 0) <= 0:
        return False
    if entry.get("check_1", "no").lower() != "yes":
        return False
    if not entry.get("rewritten_final1", ""):
        return False
    return True


# ────────────────────────── main processing ──────────────────────────

def process_entries(entries: list, strict_filter: bool) -> list:
    """
    Filter and transform entries into rows suitable for Arrow output.

    Returns a list of dicts, each representing one valid row.
    """
    valid_rows = []

    for entry in tqdm(entries, desc="Filtering entries"):
        # Basic validity checks
        if not is_valid_entry(entry):
            continue

        # Validate subject images
        img_paths = _validate_subject_images(entry, strict_filter)
        if img_paths is None:
            continue

        # Check for valid background (optional)
        bg_path, prompt2 = _validate_background(entry)

        # Build output row
        row = {
            "videoid": entry.get("videoid", ""),
            "image_name": entry.get("image_name", ""),
            "video_path": entry.get("video_path", ""),
            "prompt": entry.get("prompt", ""),
            "subject_img": json.dumps(img_paths, ensure_ascii=False),
            "prompt1": entry.get("rewritten_final1", ""),
            "background_img": bg_path if bg_path else "",
            "prompt2": prompt2 if prompt2 else "",
            "num_objects": len(img_paths),
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
        ("image_name", pa.string()),
        ("video_path", pa.string()),
        ("prompt", pa.string()),
        ("subject_img", pa.string()),
        ("prompt1", pa.string()),
        ("background_img", pa.string()),
        ("prompt2", pa.string()),
        ("num_objects", pa.int32()),
    ])

    # Build arrays
    arrays = [
        pa.array([r["videoid"] for r in rows], type=pa.string()),
        pa.array([r["image_name"] for r in rows], type=pa.string()),
        pa.array([r["video_path"] for r in rows], type=pa.string()),
        pa.array([r["prompt"] for r in rows], type=pa.string()),
        pa.array([r["subject_img"] for r in rows], type=pa.string()),
        pa.array([r["prompt1"] for r in rows], type=pa.string()),
        pa.array([r["background_img"] for r in rows], type=pa.string()),
        pa.array([r["prompt2"] for r in rows], type=pa.string()),
        pa.array([r["num_objects"] for r in rows], type=pa.int32()),
    ]

    table = pa.table(arrays, schema=schema)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with ipc.new_file(output_path, schema) as writer:
        writer.write_table(table)

    print(f"Saved {len(rows)} valid records to {output_path}")


# ────────────────────────── entry point ──────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Step9: Assemble valid results from step8 into an Arrow file")
    p.add_argument("--input_json", type=str, required=True,
                   help="Path to the step8 output JSONL file containing final "
                        "check results (check_1, check_2, etc.)")
    p.add_argument("--output", type=str, required=True,
                   help="Path to the output Arrow file")
    p.add_argument("--strict_filter", action="store_true",
                   help="Strict filtering: also requires fit == 'yes' for each "
                        "subject image. Default (lenient) only checks result == 'yes'")
    return p.parse_args()


def main():
    args = parse_args()

    # Load input data from step8 output
    all_entries = load_input_data(args.input_json)
    print(f"Loaded {len(all_entries)} entries from {args.input_json}")

    # Filter and transform
    mode_str = "strict" if args.strict_filter else "lenient"
    print(f"Filtering mode: {mode_str}")

    valid_rows = process_entries(all_entries, strict_filter=args.strict_filter)

    # Report statistics by object count
    counts = {}
    for row in valid_rows:
        n = row["num_objects"]
        counts[n] = counts.get(n, 0) + 1

    print(f"\nValid entries: {len(valid_rows)}")
    for n in sorted(counts.keys()):
        print(f"  {n} object(s): {counts[n]}")

    # Write arrow file
    write_arrow(valid_rows, args.output)


if __name__ == "__main__":
    main()
