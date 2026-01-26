import argparse
from tqdm import tqdm
import os
import json
import pickle
import copy


def write_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc=f'Writing jsonl to {filename.split("/")[-1]}'):
            # json.dumps converts dict to JSON string
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_all_pkl(folder_path):
    # obtain all pkl file paths in the folder
    pkl_files = [os.path.join(folder_path, f)
                 for f in os.listdir(folder_path)
                 if f.endswith('.pkl')]

    data_list = []

    # use tqdm for progress bar
    for file_path in tqdm(pkl_files, desc="Loading PKL Files"):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            data_list.append(data)

    return data_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='train_data/qwen25-15b-s1-32k')
    parser.add_argument('--output_path', type=str, default='train_data/qwen25-15b-s1-32k.jsonl')
    args = parser.parse_args()

    data = load_all_pkl(args.data_path)

    all_data = []
    for item in tqdm(data, desc='Processing data'):
        parsed_data = copy.deepcopy(item['original_inputs'][0])
        try:
            parsed_data['output'] = item['outputs'][0]['text']
            parsed_data['meta_info'] = item['outputs'][0]['meta_info']
        except:
            continue

        all_data.append(parsed_data)

    write_jsonl(all_data, args.output_path)
