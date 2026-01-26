import json
import os
from multiprocessing import Pool, Manager
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import requests
import pickle
import argparse


def call_model_sglang_online(query, max_new_tokens=32768, top_p=0.95):
    url = "http://127.0.0.1:30011/generate"
    data = {
        "text": query,
        "sampling_params": {"max_new_tokens": max_new_tokens,
                            "temperature": 0.6,
                            "top_p": top_p,
                            "top_k": -1,
                            "n": 1,
                            "presence_penalty": 0.0},
    }
    response = requests.post(url, json=data, timeout=7200)
    return response.json()

# Multiprocessing task
def process_chunk(datas, tokenizer, index, lock, output_dir, llm, top_p=1.0, batch_size=1):
    for j in range(0, len(datas), batch_size):
        save_path = os.path.join(output_dir, '%s_%s.pkl' % (index, j))
        if os.path.exists(save_path):
            print(f"{save_path} already exists")
            continue
        chunked_query = []
        original_inputs = []
        for line in tqdm(datas[j: j + batch_size]):
            line = json.loads(line)
            original_inputs.append(line)
            problem = line['input']

            query = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                     "Given the following message, let\'s think step by step and predict its veracity. "
                     "If it is more likely to be a real message, return \\boxed{{real}}; otherwise, return \\boxed{{fake}}: \n {problem}"
                     "<|im_end|>\n<|im_start|>assistant\n").format(
                problem=problem)

            chunked_query.append(query)

        results = call_model_sglang_online(chunked_query)
        save_dict = {"inputs": chunked_query, "original_inputs": original_inputs, "outputs": results}
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Saved to {save_path}")


def main(input_file, output_dir, tokenizer, num_processes=64, top_p=1.0, chunk_size=1):
    llm = None

    # read input data, each line is a json, only take first 10k lines
    datas = open(input_file, 'r').readlines()

    chunks = [datas[i:i + chunk_size] for i in range(0, len(datas), chunk_size)]
    chunk_i_list = []
    chunk_list = []
    for chunk_i, chunk in enumerate(chunks):
        chunk_i_list.append(chunk_i)
        chunk_list.append(chunk)

    multi_flag = True
    if multi_flag:
        with Manager() as manager:
            lock = manager.Lock()
            with Pool(processes=num_processes) as pool:
                tasks = [
                    pool.apply_async(
                        process_chunk,
                        args=(chunk, tokenizer, chunk_i, lock, output_dir, llm, top_p),
                    )
                    for chunk_i, chunk in zip(chunk_i_list, chunk_list)
                ]
                for task in tasks:
                    task.get()  # wait for all tasks to finish
    else:
        for i in range(100000000):
            process_chunk(datas, tokenizer, 0, None, output_dir, llm, top_p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str,
                        default='train_data/xmdcot-316k-id.jsonl')
    parser.add_argument("--model_path", type=str,
                        default='models/Qwen3-30B-A3B-Instruct-2507')  # you can use your local model path
    parser.add_argument("--output_path", type=str,
                        default='train_data/qwen3-a3b-instruct-s1-32k')
    parser.add_argument("--num_processes", type=int, default=96)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    main(args.input_file, args.output_path, tokenizer, num_processes=args.num_processes, top_p=args.top_p)
