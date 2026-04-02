import argparse
import csv
import os


METRIC_COLS = ["average", "min", "score_1", "score_2", "score_3"]
DEFAULT_SCORE = 1.0
TASK_TYPE = "compositional_mi2v"

DISPLAY_NAMES = {
    "average": "AVG",
    "min": "MIN",
    "score_1": "Prompt Following",
    "score_2": "Condition Preserving",
    "score_3": "Overall Visual Quality",
}


def _safe_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _process_one_csv(input_csv_path, output_csv_path, subject_name):
    """Process a single (input_csv, output_csv) pair and return (total_count, sums, missing_count, missing_indices)."""

    # ---- 1. Read input CSV ----
    with open(input_csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        input_rows = list(reader)
    total_count = len(input_rows)

    print(f"\nInput CSV ({subject_name}): {input_csv_path}")
    print(f"Total samples in input CSV: {total_count}")

    if total_count == 0:
        print(f"WARNING: Input CSV for {subject_name} is empty, skipping.")
        return 0, {col: 0.0 for col in METRIC_COLS}, 0, []

    # Build input keys
    input_keys = []
    for idx, row in enumerate(input_rows):
        key = row.get("index", str(idx))
        input_keys.append(key)

    # ---- 2. Read output CSV ----
    output_lookup = {}
    if not os.path.exists(output_csv_path):
        print(f"WARNING: Output CSV does not exist: {output_csv_path}")
        print("All samples will be treated as missing (default score = 1.0).")
    else:
        with open(output_csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for out_idx, row in enumerate(reader):
                key = row.get("index", str(out_idx))
                results_field = row.get("results", "")
                if results_field == "ERROR":
                    continue
                scores = {}
                all_valid = True
                for col in METRIC_COLS:
                    val = _safe_float(row.get(col))
                    if val is not None:
                        scores[col] = val
                    else:
                        all_valid = False
                if all_valid and scores:
                    output_lookup[key] = scores
        print(f"Output CSV ({subject_name}): {output_csv_path}")
        print(f"Valid scored samples in output CSV: {len(output_lookup)}")

    # ---- 3. Collect scores ----
    sums = {col: 0.0 for col in METRIC_COLS}
    missing_count = 0
    missing_indices = []

    for key in input_keys:
        if key in output_lookup:
            for col in METRIC_COLS:
                sums[col] += output_lookup[key][col]
        else:
            missing_count += 1
            missing_indices.append(key)
            print(f"WARNING: Sample '{key}' not found or invalid in output CSV ({subject_name}), using default score = {DEFAULT_SCORE}")
            for col in METRIC_COLS:
                sums[col] += DEFAULT_SCORE

    # ---- 4. Print per-CSV results ----
    print(f"\n{'='*60}")
    print(f"[{subject_name}] Total samples: {total_count}")
    print(f"[{subject_name}] Valid samples from output: {total_count - missing_count}")
    print(f"[{subject_name}] Missing samples: {missing_count}")
    if missing_count > 0:
        print(f"WARNING: {missing_count} samples are missing in {subject_name}, please check!")
        print(f"Missing indices: {missing_indices}")
    print(f"{'='*60}")

    print(f"\n--- Average Scores [{subject_name}] ---")
    for col in METRIC_COLS:
        avg = sums[col] / total_count
        display = DISPLAY_NAMES.get(col, col)
        print(f"  {display}: {avg:.4f}")

    return total_count, sums, missing_count, missing_indices


def main():
    parser = argparse.ArgumentParser(description="Calculate average scores for compositional_mi2v (3 CSVs: subject1/2/3).")
    parser.add_argument("--input_csv1", type=str, required=True, help="Input CSV for subject1.")
    parser.add_argument("--input_csv2", type=str, required=True, help="Input CSV for subject2.")
    parser.add_argument("--input_csv3", type=str, required=True, help="Input CSV for subject3.")
    parser.add_argument("--output_csv1", type=str, required=True, help="Output CSV for subject1.")
    parser.add_argument("--output_csv2", type=str, required=True, help="Output CSV for subject2.")
    parser.add_argument("--output_csv3", type=str, required=True, help="Output CSV for subject3.")
    args = parser.parse_args()

    print(f"Task type: {TASK_TYPE}")

    csv_configs = [
        (args.input_csv1, args.output_csv1, "subject1"),
        (args.input_csv2, args.output_csv2, "subject2"),
        (args.input_csv3, args.output_csv3, "subject3"),
    ]

    # Accumulate for weighted overall average
    global_total_count = 0
    global_sums = {col: 0.0 for col in METRIC_COLS}
    global_missing_count = 0

    for input_csv, output_csv, subject_name in csv_configs:
        total_count, sums, missing_count, missing_indices = _process_one_csv(input_csv, output_csv, subject_name)
        global_total_count += total_count
        for col in METRIC_COLS:
            global_sums[col] += sums[col]
        global_missing_count += missing_count

    # ---- Overall weighted average ----
    print(f"\n{'#'*60}")
    print(f"Overall Statistics [{TASK_TYPE}]")
    print(f"{'#'*60}")
    print(f"Total samples (all subjects): {global_total_count}")
    print(f"Valid samples (all subjects): {global_total_count - global_missing_count}")
    print(f"Missing samples (all subjects): {global_missing_count}")
    if global_missing_count > 0:
        print(f"WARNING: {global_missing_count} total samples are missing across all subjects, please check!")

    if global_total_count == 0:
        print("ERROR: No samples across all subjects, cannot compute overall average.")
        return

    print(f"\n--- Overall Weighted Average [{TASK_TYPE}] ---")
    for col in METRIC_COLS:
        avg = global_sums[col] / global_total_count
        display = DISPLAY_NAMES.get(col, col)
        print(f"  {display}: {avg:.4f}")
    print()


if __name__ == "__main__":
    main()
