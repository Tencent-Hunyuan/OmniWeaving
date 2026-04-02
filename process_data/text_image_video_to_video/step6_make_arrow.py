"""
Step6: Assemble Final Arrow Dataset

This script reads the step5 output JSONL (final consistency verification
results), filters entries that passed all checks, and assembles them into
a single Arrow file for downstream training.

For each valid entry, the Arrow record contains:
    - videoid:              unique identifier
    - condition_video_path: path to the before-edit video
    - gt_video_path:        path to the after-edit video
    - condition_img_path:   path to the extracted element image (from step2)
    - caption:              rewritten instruction (from step4)
    - instruction:          original editing instruction
    - new_element:          description of the newly introduced element

Input (--step5_json):
    A JSONL file produced by step5. Each line is a JSON object containing:
        - videoid, condition_video_path, gt_video_path,
          condition_frame_path, gt_frame_path, best_extracted_path,
          new_element, instruction, rewritten, check1, check2
    Only entries with check1 == "yes" and check2 == "yes" are included.

Output (--output):
    A single Arrow file containing all valid records.

Usage:
    python step6_make_arrow.py \\
        --step5_json /path/to/step5_output.jsonl \\
        --output /path/to/final_dataset.arrow
"""

import argparse
import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from tqdm import tqdm


# ────────────────────────── data loading ──────────────────────────

def load_valid_records(step5_json_path: str) -> list:
    """Load entries from the step5 output JSONL that passed all checks.

    Keeps only entries where:
        - check1 == "yes" and check2 == "yes"
        - rewritten is non-empty
        - best_extracted_path exists on disk
        - condition_video_path and gt_video_path exist on disk
    """
    records = []
    skipped = 0

    with open(step5_json_path, 'r', encoding='utf8') as f:
        for line in tqdm(f, desc="Loading step5 records"):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            # Filter: both checks must pass
            if entry.get('check1', '').lower() != 'yes':
                skipped += 1
                continue
            if entry.get('check2', '').lower() != 'yes':
                skipped += 1
                continue

            rewritten = entry.get('rewritten', '').strip()
            if not rewritten or rewritten == '...':
                skipped += 1
                continue

            # Verify required file paths exist
            best_extracted = entry.get('best_extracted_path', '')
            condition_video = entry.get('condition_video_path', '')
            gt_video = entry.get('gt_video_path', '')

            if not best_extracted or not os.path.exists(best_extracted):
                skipped += 1
                continue
            if not condition_video or not os.path.exists(condition_video):
                skipped += 1
                continue
            if not gt_video or not os.path.exists(gt_video):
                skipped += 1
                continue

            records.append({
                'videoid': entry['videoid'],
                'condition_video_path': condition_video,
                'gt_video_path': gt_video,
                'condition_img_path': best_extracted,
                'caption': rewritten,
                'instruction': entry.get('instruction', ''),
                'new_element': entry.get('new_element', ''),
            })

    print(f'Loaded {len(records)} valid records, skipped {skipped}')
    return records


# ────────────────────────── arrow writing ──────────────────────────

def write_arrow(records: list, output_path: str):
    """Write a list of record dicts to an Arrow IPC file."""
    if not records:
        print("No records to write.")
        return

    df = pd.DataFrame(records)
    table = pa.Table.from_pandas(df)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    writer = ipc.new_file(output_path, table.schema)
    writer.write(table)
    writer.close()

    print(f'Written {len(records)} records to {output_path}')


# ────────────────────────── entry point ──────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Step6: Assemble final Arrow dataset from step5 results")
    p.add_argument("--step5_json", type=str, required=True,
                   help="Path to step5 output JSONL file (final check results)")
    p.add_argument("--output", type=str, required=True,
                   help="Path to output Arrow file")
    return p.parse_args()


def main():
    args = parse_args()

    # Load and filter valid records
    records = load_valid_records(args.step5_json)

    # Write to Arrow
    write_arrow(records, args.output)


if __name__ == '__main__':
    main()
