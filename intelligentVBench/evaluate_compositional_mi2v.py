import argparse
import base64
import csv
import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

from tqdm import tqdm

api_key = ""
gemini_url = ""
gemini_headers = {
    "Accept": "application/json",
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}
GEMINI_MODEL = "gemini-2.5-pro"

from all_prompts import Compositional_MI2V_multi_subjects, Compositional_MI2V_1subject, Compositional_MI2V_1subject_with_background, Compositional_MI2V_multi_subjects_with_background

def extract_scores_and_average(entry: str, required_keys=None):
    import json
    import re

    score_keys = required_keys
    missing_all = list(required_keys or score_keys)
    print(required_keys)

    if not entry:
        return [], None, missing_all

    parsed = None
    try:
        parsed = json.loads(entry)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", entry, flags=re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                print(f"Error parsing JSON: {entry}")
                parsed = None
    print(f"parsed: {parsed}")
    if not isinstance(parsed, dict):
        return [], None, missing_all

    missing_keys = [key for key in score_keys if key not in parsed]
    if missing_keys:
        return [], None, missing_keys

    try:
        scores = [max(1.0, min(5.0, float(parsed[key]))) for key in score_keys]
    except (TypeError, ValueError):
        print(f"Error parsing scores: {entry}")
        return [], None, score_keys

    average = round(sum(scores) / len(scores), 2)
    return scores, average, []


def _encode_video_b64(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")


def _encode_image_b64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def call_gemini_model(
    edited_video_path,
    prompt,
    ref_image_paths=None,
    max_tokens=8192,
    temperature=0.7,
    stream=False,
):
    global gemini_headers, gemini_url

    max_retries = 6
    retry_count = 0

    while retry_count < max_retries:
        try:
            user_content = [{"type": "text", "text": prompt.strip()}]

            for i, (keyname, ref_img_path) in enumerate(ref_image_paths):
                cur_base64_image = _encode_image_b64(ref_img_path)
                numth = i + 1
                if numth == 1:
                    numth_text = "first"
                elif numth == 2:
                    numth_text = "second"
                elif numth == 3:
                    numth_text = "third"
                elif numth == 4:
                    numth_text = "fourth"

                if i == 0:
                    assert keyname == "subject_img_1"
                    user_content.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cur_base64_image}"}}
                    )
                else:
                    if keyname == "subject_img_2":
                        assert numth_text == "second"
                        user_content.append({"type": "text", "text": f"This is the {numth_text} reference image containing a specific subject:"})
                        user_content.append(
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cur_base64_image}"}}
                        )
                    elif keyname == "subject_img_3":
                        assert numth_text == "third"
                        user_content.append({"type": "text", "text": f"This is the {numth_text} reference image containing a specific subject:"})
                        user_content.append(
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cur_base64_image}"}}
                        )
                    elif keyname == "background_img":
                        assert i == len(ref_image_paths) - 1
                        user_content.append({"type": "text", "text": f"This is the {numth_text} reference image serving as the reference background image:"})
                        user_content.append(
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cur_base64_image}"}}
                        )
                    else:
                        raise ValueError(f"Invalid keyname: {keyname}")

            base64_video_after = _encode_video_b64(edited_video_path)
            user_content.append({"type": "text", "text": "This is the generated video:"})
            user_content.append(
                {"type": "image_url", "image_url": {"url": f"data:video/mp4;base64,{base64_video_after}"}}
            )

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ]

            payload = {
                "max_tokens": max_tokens,
                "messages": messages,
                "model": GEMINI_MODEL,
                "temperature": temperature,
                "stream": False
            }

            response = requests.post(gemini_url, headers=gemini_headers, data=json.dumps(payload), timeout=120)
            result = json.loads(response.text)

            if response.status_code == 200:
                if "choices" in result and result["choices"]:
                    for message in result["choices"]:
                        try:
                            message = message.get("message", {})
                            content = message.get("content", "")
                            if retry_count > 0:
                                logging.info(f"The Gemini call succeeded after {retry_count} retries.")
                            return content
                        except Exception as e:
                            logging.error(f"Error extracting content: {e}")
                            continue
                    error_msg = f"ERROR: No valid content found in choices - {result}"
                    logging.warning(f"Retry for {retry_count + 1}th time: {error_msg}")
                else:
                    error_msg = f"ERROR: No choices in response - {result}"
                    logging.warning(f"Retry for {retry_count + 1}th time: {error_msg}")
            else:
                error_msg = f"ERROR: call Gemini failed, status code: {response.status_code}, response: {result}"
                logging.warning(f"Retry for {retry_count + 1}th time: {error_msg}")

            retry_count += 1
            time.sleep(60)
        except Exception as e:
            error_msg = f"An error occurred while calling the Gemini model: {e}"
            logging.warning(f"Retry for {retry_count + 1}th time: {error_msg}")
            retry_count += 1
            time.sleep(60)

    return f"ERROR: Gemini call failed after {max_retries} retries."


def _process_single_row(row_idx, row, header, edited_video_path, max_tokens, file_parent_path=None):
    try:
        ref_img_paths = []
        multi_subject = False
        has_background = False
        prompt = row.get("prompt", "")
        for keyname in ["subject_img_1", "subject_img_2", "subject_img_3", "background_img"]:
            if row.get(keyname, ""):
                img_path = row.get(keyname)
                if file_parent_path is not None:
                    img_path = os.path.join(file_parent_path, img_path)
                assert os.path.exists(img_path)
                ref_img_paths.append((keyname, img_path))
                if keyname == "background_img":
                    has_background = True
                elif keyname == "subject_img_2" or keyname == "subject_img_3":
                    multi_subject = True

        if has_background and multi_subject:
            system_prompt = Compositional_MI2V_multi_subjects_with_background
            tt = 'subject23_back'
        elif has_background and not multi_subject:
            system_prompt = Compositional_MI2V_1subject_with_background
            tt = 'subject1_back'
        elif not has_background and multi_subject:
            system_prompt = Compositional_MI2V_multi_subjects
            tt = 'subject23'
        elif not has_background and not multi_subject:
            system_prompt = Compositional_MI2V_1subject
            tt = 'subject1'
        else:
            raise ValueError("Invalid reference images")

        full_system_prompt = system_prompt.replace("<input_prompt>", prompt)

        filename = row.get("index", "")+'.mp4'
        edited_result_path = os.path.join(edited_video_path, os.path.basename(filename))

        if not os.path.exists(edited_result_path):
            original_row = [row.get(col, "") for col in header]
            return (
                row_idx,
                original_row
                + [
                    f"ERROR",
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                1,
            )

        response = call_gemini_model(
            edited_result_path,
            full_system_prompt,
            ref_image_paths=ref_img_paths,
            max_tokens=max_tokens,
        )
        formatted_response = response.replace("\n", "\\n")
        print('formatted_response: ', formatted_response)
        required_keys_by_type = {
            "subject1_back": [
                "Prompt Following",
                "Subject and Background Consistency",
                "Overall Visual Quality",
            ],
            "subject23": [
                "Prompt Following",
                "Multi-Subject Consistency",
                "Overall Visual Quality",
            ],
            "subject1": [
                "Prompt Following",
                "Subject Consistency",
                "Overall Visual Quality",
            ],
            "subject23_back": [
                "Prompt Following",
                "Multi-Subject and Background Consistency",
                "Overall Visual Quality",
            ],
        }
        required_keys = required_keys_by_type.get(tt, [])
        scores, average_score, missing_keys = extract_scores_and_average(
            response, required_keys=required_keys
        )
        if missing_keys:
            raise ValueError(
                f"Missing required fields for {tt}: {', '.join(missing_keys)}"
            )
        score_1 = scores[0] if len(scores) > 0 else None
        score_2 = scores[1] if len(scores) > 1 else None
        score_3 = scores[2] if len(scores) > 2 else None
        min_score = (
            min(s for s in [score_1, score_2, score_3] if s is not None)
            if any(s is not None for s in [score_1, score_2, score_3])
            else None
        )
        original_row = [row.get(col, "") for col in header]

        return (
            row_idx,
            original_row
            + [formatted_response, average_score, min_score, score_1, score_2, score_3],
            average_score,
        )
    except Exception as e:
        original_row = [row.get(col, "") for col in header]
        print(f"Error processing row: {row_idx}")
        print(f"Error: {e}")
        return (
            row_idx,
            original_row + [f"ERROR", "ERROR", "", "", "", ""],
            None,
        )


def process_csv(
    input_csv_path, output_csv_path, edited_video_path, num_threads=16, file_parent_path=None
):

    with open(input_csv_path, "r", encoding="utf-8-sig") as infile:
        reader = csv.DictReader(infile)
        header = reader.fieldnames
        rows = list(reader)

    existing_results = {}
    if os.path.exists(output_csv_path):
        with open(output_csv_path, "r", encoding="utf-8-sig") as existing_file:
            existing_reader = csv.DictReader(existing_file)
            for row in existing_reader:
                if not header:
                    continue
                key = tuple(row.get(col, "") for col in header)
                results_field = row.get("results", "")
                average_field = row.get("average", "")
                min_field = row.get("min", "")
                score_1_field = row.get("score_1", "")
                score_2_field = row.get("score_2", "")
                score_3_field = row.get("score_3", "")
                if min_field and average_field and results_field != "ERROR":
                    try:
                        average_score = float(average_field)
                    except ValueError:
                        average_score = None
                    output_row = [
                        row.get(col, "") for col in header
                    ] + [
                        results_field,
                        average_field,
                        min_field,
                        score_1_field,
                        score_2_field,
                        score_3_field,
                    ]
                    existing_results[key] = (output_row, average_score)
    print('output_csv_path: ', output_csv_path)
    print('existing_results length: ', len(existing_results))

    results = [None] * len(rows)
    for row_idx, row in enumerate(rows):
        key = tuple(row.get(col, "") for col in header)
        if key in existing_results:
            results[row_idx] = existing_results[key]
    

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for row_idx, row in enumerate(rows):
            if results[row_idx] is not None:
                continue
            futures.append(
                executor.submit(
                    _process_single_row,
                    row_idx,
                    row,
                    header,
                    edited_video_path,
                    8192,
                    file_parent_path,
                )
            )

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {os.path.basename(input_csv_path)}"):
            row_idx, output_row, average_score = future.result()
            if output_row is not None and average_score is not None:
                results[row_idx] = (output_row, average_score)

    with open(output_csv_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile, delimiter=",")
        writer.writerow(
            header + ["results", "average", "min", "score_1", "score_2", "score_3"]
        )

        for result in results:
            if result is None:
                continue
            output_row, average_score = result
            writer.writerow(output_row)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use the Gemini 2.5 Pro model to process video and prompt CSV files.")
    parser.add_argument(
        "--input_csv1",
        type=str,
        default="",
        help="The path to the subject1 input CSV file.",
    )
    parser.add_argument(
        "--input_csv2",
        type=str,
        default="",
        help="The path to the subject2 input CSV file.",
    )
    parser.add_argument(
        "--input_csv3",
        type=str,
        default="",
        help="The path to the subject3 input CSV file.",
    )
    parser.add_argument(
        "--edit_path1",
        type=str,
        default="",
        help="The path to the subject1 edited video results.",
    )
    parser.add_argument(
        "--edit_path2",
        type=str,
        default="",
        help="The path to the subject2 edited video results.",
    )
    parser.add_argument(
        "--edit_path3",
        type=str,
        default="",
        help="The path to the subject3 edited video results.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory.",
    )
    parser.add_argument("--num_threads", type=int, default=50, help="Number of threads for parallel Gemini calls.")
    parser.add_argument("--file_parent_path", type=str, default=None, help="Parent path to prepend to relative image/video paths in the CSV.")
    args = parser.parse_args()

    csv_configs = [
        (args.input_csv1, args.edit_path1, "subject1"),
        (args.input_csv2, args.edit_path2, "subject2"),
        (args.input_csv3, args.edit_path3, "subject3"),
    ]

    for input_csv, edit_path, subject_name in csv_configs:
        if not input_csv or not edit_path:
            print(f"Skipping {subject_name}: input_csv or edit_path not provided.")
            continue

        output_dir = args.output_dir if args.output_dir else os.path.join(edit_path, "metrics")
        cur_output_dir = os.path.join(output_dir, subject_name)
        os.makedirs(cur_output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(input_csv))[0]
        output_csv = os.path.join(cur_output_dir, f"{GEMINI_MODEL}_{base_name}.csv")

        print(f"\n{'='*60}")
        print(f"Processing {subject_name}: {input_csv}")
        print(f"Edit path: {edit_path}")
        print(f"Output CSV: {output_csv}")
        print(f"{'='*60}")

        process_csv(
            input_csv_path=input_csv,
            output_csv_path=output_csv,
            edited_video_path=edit_path,
            num_threads=args.num_threads,
            file_parent_path=args.file_parent_path,
        )
