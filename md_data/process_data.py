"""
processing original data into Dict{'text', 'label'}
Data Source:
GossipCop++ / PolitiFact++: https://github.com/mbzuai-nlp/Fakenews-dataset
MultiDis / EUDisinfo: https://github.com/ArkadiusDS/PCoT
"""
import argparse
import json
import pandas as pd

def process_fakenewsnet(real_path, fake_path, output_path):
    real_data = json.load(open(real_path))
    fake_data = json.load(open(fake_path))

    real_data = [real_data[str(i)] for i in range(min(500, len(real_data)))]
    fake_data = [fake_data[str(i)] for i in range(min(500, len(fake_data)))]
    # dict_keys(['id', 'text', 'title', 'description'])

    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in real_data:
            item['label'] = 'real'
            count += 1
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
        for item in fake_data:
            item['label'] = 'fake'
            count += 1
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'the processed data has been saved to {output_path}, total {count} items')
    return count

def process_pcot(path, output_path):
    df = pd.read_csv(path, encoding='utf-8')
    # ['Article_ID', 'label', 'Article_Topic', 'Article_Publication_Date', 'content']
    process = []
    for index, row in df.iterrows():
        process.append({
            'id': row.get('Article_ID', index),
            'text': row['content'],
            'label': row['label'],
        })
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in process:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'the processed data has been saved to {output_path}, total {len(process)} items')
    return len(process)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='multidis',
                        help='gossipcop, politifact, multidis, eudisinfo')
    args = parser.parse_args()
    if args.dataset == 'gossipcop':
        num = process_fakenewsnet(real_path='gossipcop_real.json', fake_path='gossipcop_fake.json',
                                  output_path='processed_gossipcop.jsonl')
    elif args.dataset == 'politifact':
        num = process_fakenewsnet(real_path='politifact_real.json', fake_path='politifact_fake.json',
                                  output_path='processed_politifact.jsonl')
    elif args.dataset == 'multidis':
        num = process_pcot(path='multidis.csv', output_path='processed_multidis.jsonl')
    elif args.dataset == 'eudisinfo':
        num = process_pcot(path='eudisinfo.csv', output_path='processed_eudisinfo.jsonl')
