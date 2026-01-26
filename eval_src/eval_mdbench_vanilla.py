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


def call_model(query, client, model):
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


# Check processed ID set (for resume)
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
def process_data(data, client, model, sample_num=1):
    text = data['text']
    template = (
        'Given the following message, predict its veracity. If it is more likely to be a real message, return \\boxed{{real}}";'
        ' otherwise, return \\boxed{{fake}}. Please refrain from providing ambiguous assessments such as undetermined: \n {problem}')
    try:
        query = template.format(problem=text)
        answer, output_token_length, input_token_length = call_model(query, client, model)
        data['answer'] = answer
        data['input_token_length'] = input_token_length
        data['output_token_length'] = output_token_length

    except Exception as e:
        print("[Exception type]:", type(e).__name__)
        # traceback.print_exc()  # Print standard error stack trace

    return data


# Multiprocessing task handler
def process_chunk(chunk, chunk_i, processed_ids, lock, ip, output_file, model, sample_num, api_key):
    if chunk_i in processed_ids:  # Skip already processed data
        return
    client = openai.Client(base_url=ip, api_key=api_key)
    results = []
    for n in range(sample_num):
        for item in chunk:
            result = process_data(item, client, model, sample_num)
            results.append(result)
        write_results(results, output_file, chunk_i, n)
        print(f"---->>> data {chunk_i}-{n}.jsonl written to {output_file}")


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

            # Launch multiprocessing pool
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
                    task.get()  # wait for all tasks to finish

        # 2. Merge and clean up result files
        results = []
        with open(output_path, "w", encoding="utf-8") as outfile:
            # Iterate temporary files (only .jsonl)
            for filename in os.listdir(temp_path):
                if filename.endswith(".jsonl"):
                    filepath = os.path.join(temp_path, filename)
                    with open(filepath, "r", encoding="utf-8") as infile:
                        line = infile.readline().strip()
                        results.append(json.loads(line))
                        if line:  # guard against empty lines
                            outfile.write(line + "\n")
        shutil.rmtree(temp_path)

        print(f"Merge complete. Output file: {output_path}; temporary directory {temp_path} removed")

    # 3. Evaluate outputs
    y_true = []  # true labels
    y_pred = []  # predicted labels

    for result in results:
        answer = result['answer']
        label = result['label']

        if answer is not None:
            # assumes 'real' label -> 1, others -> 0
            y_true.append(1 if label == 'real' else 0)  # define mapping here
            # prediction rule: answer containing 'real' indicates positive class
            y_pred.append(1 if 'real' in answer else 0)
        else:
            continue

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # F1 scores (per class)
    f1_pos = f1_score(y_true, y_pred, pos_label=1)  # positive F1
    f1_neg = f1_score(y_true, y_pred, pos_label=0)  # negative F1

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
