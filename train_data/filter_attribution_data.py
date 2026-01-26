import argparse
from tqdm import tqdm
import json
import math
import numpy as np
import re
from collections import defaultdict


def read_json(path):
    data = []
    with open(path, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc=f'Reading jsonl from {path}'):
            data.append(json.loads(line))
    return data


def find_last_empty_step(steps):
    """
    Find from the end the first item whose step_text is an empty string.
    :param steps: list[dict]
    :return: (index, item) if found, else (None, None)
    """
    for idx in range(len(steps) - 1, -1, -1):
        if steps[idx].get("step_text", "") == "":
            return idx, steps[idx]
    return None, None


def extract_boxed(text: str):
    """
    Extract the content wrapped by boxed{...}. Return None if not found.
    """
    match = re.search(r'boxed\{(.*?)\}', text)
    if match:
        return match.group(1)
    return None


def has_excessive_in_step_repeats(text, threshold=2, step_delimiter='\n\n', in_step_delimiter='\n', cut_length=0):
    # Split the text into steps
    text = text[cut_length:]
    steps = text.split(step_delimiter)

    for step in steps:
        in_step_counts = defaultdict(int)
        in_steps = step.split(in_step_delimiter)
        prev = in_steps[0]
        count = 0
        for index, in_step in enumerate(in_steps):
            if in_step in ["\\[", "\\[ ", "\\]", "\\] ", "\\(", "\\( ", "\\)", "\\) ", "\\{", "\\{ ", "\\}", "\\} ",
                           "\\text{", "\\text{ ", "\\begin{", "\\begin{ ", "\\end{", "\\end{ ",
                           "\\", "\\ ", " ", "  ", "   ", "    ", "      ",
                           "This simplifies to:"]:
                prev = in_step
                continue
            if index != 0:
                if in_step == prev:
                    count += 1
                    if count > threshold:
                        return True
                else:
                    count = 0
                prev = in_step
    for index, step_i in enumerate(steps):
        # has_repeat = bool(re.search(r'(.+)\1', step_i))
        if "-----------" in step_i or "           " in step_i:
            continue
        pattern = re.search(r'(.{10,})\1\1\1', step_i)
        has_repeat = bool(pattern)
        if has_repeat:
            # pdb.set_trace()
            return True

    return False


def has_excessive_step_repeats(text, threshold=10, delimiter='\n\n'):
    # Split the text into steps
    steps = text.split(delimiter)
    num_steps = len(steps)

    # Count frequency of each step
    step_counts = defaultdict(int)
    continous_repeat_count = 0
    prev_step = steps[0]
    for index, step_i in enumerate(steps):
        if index != 0:
            if step_i == prev_step:
                continous_repeat_count += 1
                if continous_repeat_count > 2:
                    return 1
            else:
                continous_repeat_count = 0
            prev_step = step_i

        step_counts[step_i] += 1

    sorted_keys = sorted(step_counts, key=lambda x: step_counts[x], reverse=True)
    for index, key in enumerate(sorted_keys):
        if step_counts[key] > threshold:
            if len(key.split(" ")) > 6:
                return 2
            elif len(key.split(" ")) > 3 and step_counts[key] > 30:
                return 2
            elif len(key.split(" ")) > 1 and step_counts[key] > 50:
                return 2
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str,
                        default='/path/to/output/attribution-{}k.jsonl')
    parser.add_argument('--top_k', type=int, default=200000)
    args = parser.parse_args()

    args.input_file = [
        '/path/to/input/qwen3-32b-s1-32k_qwen3_4b_scores.jsonl',
        '/path/to/input/qwen3-a3b-instruct-s1-32k_qwen3_4b_scores.jsonl',
        '/path/to/input/qwen3-next-a3b-s1-32k_qwen3_4b_scores.jsonl',
    ]
    data = []
    for input_file in args.input_file:
        data.extend(read_json(input_file))

    results = []
    lengths = []

    # Iterate over data to compute metrics and lengths
    for item in tqdm(data, desc='Processing jsonl'):
        label = item['label']
        input_text = item['input']
        output_field = next((k for k in item.keys() if 'output' in k.lower()), None)
        output = item[output_field]

        # filter incorrect ones
        answer = extract_boxed(output.split('\n\n')[-1])
        if answer is None:
            continue
        if item['label'] not in answer:  # answer = output.split('\n\n')[-1]
            continue
        flag = has_excessive_step_repeats(output)
        if flag == 1 or flag == 2 or has_excessive_in_step_repeats(output):
            continue

        _, full_rationale = find_last_empty_step(item['steps_scores'])
        try:
            full_score = math.log(full_rationale['probabilities'][label])
        except:
            full_score = -100
            print(f"math log error for the value {full_rationale['probabilities'][label]}")

        deltas = []
        recs = []
        for rationale in item['steps_scores'][:-1]:
            try:
                score = math.log(rationale['probabilities'][label])
            except:
                score = -100
                print(f"math log error for the value {rationale['probabilities'][label]}")

            deltas.append(full_score - score)
            recs.append(1 if full_score - score > 0 else 0)

        metric = (sum(deltas) / len(deltas)) * (1 - sum(recs) / len(deltas))
        length = len(item['steps_scores'])

        lengths.append(length)
        results.append({
            "metric": metric,
            "length": length,
            "input": input_text,
            "output": output,
            "label": label
        })

    print(f"total number of filtered samples: {len(results)}")
    print(f"average rationales length: {sum(lengths) / len(lengths)}")

    lengths_np = np.array(lengths)
    lower_bound = int(np.percentile(lengths_np, 10))
    upper_bound = int(np.percentile(lengths_np, 90))

    print(f"Auto-selected length range: {lower_bound} ~ {upper_bound}")

    # Select samples within the acceptable length range
    results = [r for r in results if lower_bound <= r["length"] <= upper_bound]

    # Sort by metric in descending order
    results.sort(key=lambda x: x["metric"], reverse=True)

    # Keep the top_k records
    selected = results[:args.top_k]

    # Write output jsonl
    with open(args.output_file.format(len(selected) // 1000), 'w', encoding='utf-8') as f:
        for r in tqdm(selected, desc='writing jsonl'):
            record = {
                "input": r["input"],
                "output": r["output"],
                "label": r["label"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Total samples: {len(data)}, samples within range: {len(results)}, saved: {len(selected)} samples to {args.output_file}")
