import os
import argparse
import json
import re
import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict


def read_jsonl(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        data = []
        with open(path, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc=f'reading {path.split("/")[-1]}'):
                data.append(json.loads(line))
    return data


def extract_boxed(text: str):
    """
    Extract the content wrapped by boxed{...} from a string.
    Returns None if no match is found.
    """
    match = re.search(r'boxed\{(.*?)\}', text)
    if match:
        return match.group(1)
    return None


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


def plot_distribution(data1, data2, img_filename, bin, min_perc=10, max_perc=80):
    """
    Plot histograms for two data lists and save as Excel.
    data1, data2: list or numpy.array, raw data
    img_filename: str, histogram output filename
    bin: number of bins
    min_perc, max_perc: percentile clipping range
    """
    base, _ = os.path.splitext(img_filename)
    excel_filename = base + ".xlsx"

    # Convert to numpy arrays
    arr1 = np.array(data1)
    arr2 = np.array(data2)

    # Compute percentile bounds and clip the values
    lower1 = np.percentile(arr1, min_perc)
    upper1 = np.percentile(arr1, max_perc)
    filtered1 = arr1[(arr1 >= lower1) & (arr1 <= upper1)]

    lower2 = np.percentile(arr2, min_perc)
    upper2 = np.percentile(arr2, max_perc)
    filtered2 = arr2[(arr2 >= lower2) & (arr2 <= upper2)]

    print(f"Data1 range: {lower1:.4f} - {upper1:.4f}")
    print(f"Data2 range: {lower2:.4f} - {upper2:.4f}")

    # Use combined range from both datasets to define bin edges
    combined_min = min(filtered1.min(), filtered2.min())
    combined_max = max(filtered1.max(), filtered2.max())
    bins = np.linspace(combined_min, combined_max, bin + 1)

    # Draw overlapping histograms with distinct colors
    plt.figure(figsize=(8, 5))
    plt.hist(filtered1, bins=bins, alpha=0.5, color='blue', label='Data 1', edgecolor='black')
    plt.hist(filtered2, bins=bins, alpha=0.5, color='orange', label='Data 2', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Distribution (Percentiles {min_perc} - {max_perc})')
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig(img_filename, dpi=600)
    plt.close()

    # Persist histogram data to Excel
    counts1, _ = np.histogram(filtered1, bins=bins)
    counts2, _ = np.histogram(filtered2, bins=bins)
    df = pd.DataFrame({
        'Bin Start': bins[:-1],
        'Bin End': bins[1:],
        'Count Data1': counts1,
        'Count Data2': counts2
    })
    df.to_excel(excel_filename, index=False)

    print(f"Image saved to {img_filename}")
    print(f"Excel data saved to {excel_filename}")


def plot_unnecessary_ratio(data, output_path_unnecess, output_path_deltas, epsilon=0.005):
    correct_unnecessary_ratio = []
    correct_deltas_list = []
    incorrect_unnecessary_ratio = []
    incorrect_deltas_list = []
    # Iterate data to compute metrics and lengths
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
            label_pred = 'fake' if label == 'real' else 'real'
        else:
            label_pred = label

        _, full_rationale = find_last_empty_step(item['steps_scores'])
        try:
            full_score = math.log(full_rationale['probabilities'][label_pred])
        except:
            continue

        deltas = []
        unnecess = []
        for rationale in item['steps_scores'][:-1]:
            try:
                score = math.log(rationale['probabilities'][label_pred])
            except:
                continue

            if full_score - score == 0:
                continue

            if item['label'] in answer:
                correct_deltas_list.append(full_score - score)
            else:
                incorrect_deltas_list.append(full_score - score)
            unnecess.append(1 if full_score - score < epsilon else 0)
        if len(unnecess) == 0:
            continue
        avg_unnecess = sum(unnecess) / len(unnecess)
        if item['label'] in answer:
            correct_unnecessary_ratio.append(avg_unnecess)
        else:
            incorrect_unnecessary_ratio.append(avg_unnecess)

    print(f"len(correct_unnecessary_ratio) = {len(correct_unnecessary_ratio)}")
    print(f"len(incorrect_unnecessary_ratio) = {len(incorrect_unnecessary_ratio)}")
    print(f"len(correct_deltas_list) = {len(correct_deltas_list)}")
    print(f"len(incorrect_deltas_list) = {len(incorrect_deltas_list)}")
    plot_distribution(correct_deltas_list, incorrect_deltas_list, output_path_deltas, bin=25, min_perc=10, max_perc=90)
    plot_distribution(correct_unnecessary_ratio, incorrect_unnecessary_ratio, output_path_unnecess, bin=25, min_perc=10, max_perc=90)


def statistic_insufficient(data, epsilon=0.005):
    count_insufficient = defaultdict(int)
    deltas = []
    # Iterate data to compute metrics and lengths
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
            label_pred = 'fake' if label == 'real' else 'real'
        else:
            label_pred = label

        _, full_rationale = find_last_empty_step(item['steps_scores'])
        try:
            full_score = math.log(full_rationale['probabilities'][label_pred])
        except:
            continue

        necess = []
        delta = []
        for rationale in item['steps_scores'][:-1]:
            try:
                score = math.log(rationale['probabilities'][label_pred])
            except:
                continue

            if full_score - score == 0:
                continue

            necess.append(1 if full_score - score > epsilon else 0)
            delta.append(full_score - score)
        if len(necess) == 0:
            continue
        if item['label'] in answer:
            count_insufficient[sum(necess)] += 1
            deltas.extend(delta)
    # print results
    print(f"epsilon = {epsilon}")
    print(f"average delta = {sum(deltas) / len(deltas)}")
    total = sum(count_insufficient.values())
    for num in sorted(count_insufficient)[:10]:
        ratio = count_insufficient[num] / total
        print(f"containing {num} sufficient rationales: {count_insufficient[num]} ({ratio:.2%})")


def statistic_step_number(data, output_path):
    correct_step_number = []
    incorrect_step_number = []
    for item in tqdm(data, desc='Processing jsonl'):
        output_field = next((k for k in item.keys() if 'output' in k.lower()), None)
        output = item[output_field]
        answer = extract_boxed(output.split('\n\n')[-1])
        if answer is None:
            continue

        number = len(output.split("\n\n"))
        if item['label'] in answer:
            correct_step_number.append(number)
        else:
            incorrect_step_number.append(number)

    print(f"correct_step_number = {len(correct_step_number)}")
    print(f"incorrect_step_number = {len(incorrect_step_number)}")
    plot_distribution(correct_step_number, incorrect_step_number, output_path, bin=25, min_perc=0, max_perc=95)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", type=str, default="./train_data")
    parser.add_argument("--epsilon", type=float, default=0.00)
    parser.add_argument("--data_path", type=str, default="qwen3-a3b-instruct-s1-32k_qwen3_4b_scores.jsonl")
    args = parser.parse_args()

    data_path = os.path.join(args.task_dir, args.data_path)
    args.output_dir = os.path.join(args.task_dir, 'visualization')

    data = read_jsonl(data_path)
    plot_unnecessary_ratio(data,
                           output_path_unnecess=os.path.join(args.output_dir, 'mixed_unnecessary_ratio_' + args.data_path.replace(".jsonl", ".jpg")),
                           output_path_deltas=os.path.join(args.output_dir, 'mixed_deltas_' + args.data_path.replace(".jsonl", ".jpg")))
    statistic_insufficient(data, epsilon=args.epsilon)
    statistic_step_number(data, output_path=os.path.join(args.output_dir, 'step_number_' + args.data_path.replace(".jsonl", ".jpg")))
