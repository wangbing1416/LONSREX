import argparse
import os
import json
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from typing import List, Dict, Any, Optional, Callable
from multiprocessing import Pool, Manager
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sglang as sgl
import openai
import traceback
import shutil
from sklearn.metrics import accuracy_score, f1_score
import re


def call_model_think(query, client, model):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": query
            }
        ],
        stream=False,
        temperature=0.6,
        top_p=0.95,
        max_tokens=32768,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": True},
        },
        timeout=1200
    )
    return response.choices[0].message.content, response.choices[0].message.reasoning_content, response.usage.completion_tokens, response.usage.prompt_tokens


def call_model_no_think(query, client, model, system=None):
    message = [{"role": "user", "content": query}]
    if system:
        message.append({"role": "system", "content": system})
    response = client.chat.completions.create(
        model=model,
        messages=message,
        stream=False,
        temperature=0.6,
        top_p=0.95,
        max_tokens=4096,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=1200
    )
    return response.choices[0].message.content, response.usage.completion_tokens, response.usage.prompt_tokens


def read_json(path):
    data = []
    try:
        with open(path, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc=f'processing {path.split("/")[-1]}'):
                data.append(json.loads(line))
    except:
        print(f'failed to read {path.split("/")[-1]}')
    return data


# Write processed results to file
def write_results(result, output_file, chunk_i, n):
    with open(os.path.join(output_file, f"{chunk_i}-{n}.jsonl"), "w", encoding="utf-8") as f:
        for line in result:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def group_by_field(data: List[Dict[str, Any]], field: str, keep_order: bool = True) -> List[List[Dict[str, Any]]]:
    grouped = OrderedDict() if keep_order else defaultdict(list)

    for item in data:
        if field not in item:
            raise KeyError(f"Field '{field}' not found in: {item}")
        key = item[field]

        if keep_order:
            # Initialize the group if the key is not seen before
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(item)
        else:
            grouped[key].append(item)
    return list(grouped.values())


# Retrieve processed chunk identifiers (for resuming)
def get_processed_ids(output_file):
    if not os.path.exists(output_file):
        return set(), list()
    res = set()
    existing_files = os.listdir(output_file)
    candidate_files = []
    for existing_file in existing_files:
        if existing_file.endswith('jsonl'):
            candidate_files.append(int(existing_file.split('.')[0]))
    return res, candidate_files


# Simulated data processing function
def process_data(data, client, model, sample_num=1, output_file=None):
    text = data['text']
    template = (
        'Given the following message, let\'s think step by step and predict its veracity. '
        'If it is more likely to be a real message, return \\boxed{{real}}; otherwise, return \\boxed{{fake}}: \n {problem}')

    try:
        first_query = template.format(problem=text)
        first_answer, first_think, _, _ = call_model_think(first_query, client, model)
    except Exception as e:
        print("[Exception type]:", type(e).__name__)
        traceback.print_exc()  # Print standard error stack trace info
        first_answer, first_think = None, None

    correct_template = ('You will be given: \n A Message to predict its veracity: {problem} \n'
        'A Reasoning Trace from another model regarding the decision of this message: {think} \n\n'
        'Your tasks:\n'
        'Step 1: Assess whether the Reasoning Trace is logically sound, factually correct, and coherent in the context of the given Message. \n'
        'Step 2: If the Reasoning Trace is entirely correct, return it exactly as provided, with no modifications.\n'
        'If the Reasoning Trace contains errors, missing details, or flawed logic, extract the knowledge elements it references, '
        'correct and expand them as needed, then rewrite the Reasoning Trace using the corrected and enriched knowledge.\n\n'
        'Important: Your output must contain only the final Reasoning Trace (either original or rewritten), with no additional commentary, explanation, or meta-text.\n'
        'Format: <output> Final Reasoning Trace </output> ')

    try:
        correct_query = correct_template.format(problem=text, think=first_think)
        correct_think, _, _ = call_model_no_think(correct_query, client, model)
        match = re.search(r"<output>(.*?)</output>", correct_think)
        if match:
            correct_think = match.group(1).strip()  # 去掉前后的空格
    except Exception as e:
        print("[Exception type]:", type(e).__name__)
        traceback.print_exc()  # Print standard error stack trace info
        correct_think = None

    if 'gemma' in output_file:
        detec_system = None
    else:
        detec_system = 'You are an AI assistant trained to detect fake news.'
    detec_template = (
        'Given the following message, and its step by step think process. \n'
        'Please predict its veracity. '
        'If it is more likely to be a real message, return \\boxed{{real}}; otherwise, return \\boxed{{fake}}. \n'
        'Message: {problem}, Think: {think}')

    try:
        query = detec_template.format(problem=text, think=correct_think)
        answer, output_token_length, input_token_length = call_model_no_think(query, client, model, system=detec_system)
    except Exception as e:
        print("[Exception type]:", type(e).__name__)
        traceback.print_exc()  # Print standard error stack trace info
        answer, output_token_length, input_token_length = None, None, None

    data['first_answer'] = first_answer
    data['first_think'] = first_think
    data['correct_think'] = correct_think
    data['answer'] = answer
    data['input_token_length'] = input_token_length
    data['output_token_length'] = output_token_length

    return data


# Multi-process task handler
def process_chunk(chunk, chunk_i, processed_ids, lock, ip, output_file, model, sample_num, api_key):
    if chunk_i in processed_ids:  # Skip already processed data
        return
    client = openai.Client(base_url=ip, api_key=api_key)
    results = []
    for n in range(sample_num):
        for item in chunk:
            result = process_data(item, client, model, sample_num, output_file)
            results.append(result)
        write_results(results, output_file, chunk_i, n)
        print(f"---->>> data {chunk_i}_{n}.jsonl written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='politifact',
                        help='gossipcop, politifact, multidis, eudisinfo')
    parser.add_argument('--chunk_size', type=int, default=1, help="chunk size")
    parser.add_argument('--num_processes', type=int, default=128, help="number of multiprocessing processes")
    parser.add_argument('--sample_num', type=int, default=4, help="number of samples per question")
    parser.add_argument('--output_path', type=str, default='eval_qwen3_4b_instruct')
    parser.add_argument('--model', type=str, default='', help="model path or id; prefer MODEL_PATH environment variable")
    parser.add_argument('--ip', type=str, default='http://127.0.0.1:30011/v1')
    args = parser.parse_args()

    args.data_path = f'md_data/processed_{args.dataset}.jsonl'
    data = read_json(args.data_path)

    temp_path = f'eval_results/{args.output_path}/{args.dataset}_n{args.sample_num}'
    output_path = f'eval_results/{args.output_path}/{args.dataset}_outputs_n{args.sample_num}.jsonl'
    results_path = f'eval_results/{args.output_path}/{args.dataset}_results_n{args.sample_num}.json'

    if os.path.exists(output_path):
        results = read_json(output_path)
    else:
        os.makedirs(temp_path, exist_ok=True)
        _, processed_chunk_ids = get_processed_ids(temp_path)
        print(f"---->>> processed_chunk_ids: {processed_chunk_ids}")

        chunks = [data[i:i + args.chunk_size] for i in range(0, len(data), args.chunk_size)]
        with Manager() as manager:
            lock = manager.Lock()
            shared_processed_ids = manager.list(processed_chunk_ids)

            # Start multiprocessing pool
            with Pool(processes=args.num_processes) as pool:
                tasks = [
                    pool.apply_async(
                        process_chunk,
                        args=(chunk, chunk_i, shared_processed_ids,
                              lock, args.ip, temp_path, args.model, args.sample_num),
                    )
                    for chunk_i, chunk in enumerate(chunks)
                ]
                for task in tasks:
                    task.get()  # Wait for all tasks to finish

        # Merge and clean up temporary files
        results = []
        with open(output_path, "w", encoding="utf-8") as outfile:
            # Iterate through temp folder files (only .jsonl)
            for filename in os.listdir(temp_path):
                if filename.endswith(".jsonl"):
                    filepath = os.path.join(temp_path, filename)
                    with open(filepath, "r", encoding="utf-8") as infile:
                        line = infile.readline().strip()
                        results.append(json.loads(line))
                        if line:  # Avoid empty lines
                            outfile.write(line + "\n")
        shutil.rmtree(temp_path)

        print(f"Merge completed. Output file: {output_path}. Temporary folder {temp_path} has been removed.")

    # Evaluate outputs section
    y_true = []  # True labels
    y_pred = []  # Predicted labels

    for result in results:
        answer = result['answer']
        if answer is None:
            answer = result['think'].split('\n\n')[-1]
        if 'gptoss' in args.output_path:
            answer = answer.split('<|channel|>final<|message|>')[-1]
        label = result['label']
        if answer is not None:
            # Treat the ground truth label "real" as positive (1) and other labels as negative (0).
            y_true.append(1 if label == 'real' else 0)
            # Prediction rule: answers containing 'real' are classified as positive.
            y_pred.append(1 if 'real' in answer else 0)
        else:
            continue

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # F1 per class
    f1_pos = f1_score(y_true, y_pred, pos_label=1)
    f1_neg = f1_score(y_true, y_pred, pos_label=0)

    # Micro / Macro F1
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    metrics = {
        "dataset": args.dataset,
        "accuracy": acc,
        "f1_positive": f1_pos,
        "f1_negative": f1_neg,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "real_label_num": int(sum(y_true)),
        "fake_label_num": int(len(y_true) - sum(y_true)),
    }

    print(f"DATASET: {args.dataset}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (positive): {f1_pos:.4f}")
    print(f"F1 (negative): {f1_neg:.4f}")
    print(f"F1 (micro): {f1_micro:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print(f"Metrics saved to {results_path}")
