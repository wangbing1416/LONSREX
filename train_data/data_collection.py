import argparse
import json
import logging
from tqdm import tqdm
from datasets import load_dataset


def write_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc=f'Writing jsonl to {filename.split("/")[-1]}'):
            # json.dumps converts dict to JSON string
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_md_datasets(dataset_name, cache, text_field, label_field, real, fake):
    dataset = load_dataset(dataset_name, cache_dir=cache)
    split = dataset.keys()
    dataset_data = []
    for s in split:
        dataset_data.extend(dataset[s])
    clean_data = []
    for d in tqdm(dataset_data, desc=f'Loading {dataset_name}'):
        label = d[label_field]
        input = d[text_field]
        # custom some datasets
        if 'Hasib18' in dataset_name:
            input = input.split("Input:")[1].split("Output:")[0].strip()
        if 'argilla' in dataset_name:
            label = d[label_field][0]['label']

        if label == real:
            gt = 'real'
        elif label == fake:
            gt = 'fake'
        else:
            continue
        clean_data.append({
            'input': input, 'label': gt, 'source': dataset_name
        })
    return clean_data

def check_overlap_and_merge(all_data, loaded_data, overlap_char=100):
    # 1. Build a set of the first 100 characters for all_data
    # 2. Count overlaps & collect non-overlapping items
    # 3. Merge non-overlapping items into all_data
    all_prefixes = {item["input"][:overlap_char].lower() for item in all_data}

    overlap_count = 0
    non_overlap_items = []
    for item in tqdm(loaded_data, desc='Checking overlap'):
        prefix = item["input"][:overlap_char].lower()
        if prefix in all_prefixes:
            overlap_count += 1
        else:
            non_overlap_items.append(item)

    all_data.extend(non_overlap_items)
    # output results
    logger.info(f"overlap number: {overlap_count}, total loaded number: {len(loaded_data)}")
    logger.info(f"overlap rate: {overlap_count / len(loaded_data)}")
    logger.info(f"number of all_data after checking: {len(all_data)}")
    return all_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='train_data/train_data_{num}.jsonl')
    parser.add_argument('--log_path', type=str, default='train_data/logging_{char}.log')
    parser.add_argument('--overlap_char', type=int, default=100)
    args = parser.parse_args()

    args.log_path = args.log_path.format(char=args.overlap_char)

    all_data = []
    # (dataset_name, cache, text_field, label_field, real, fake)
    meta_info = [
        ('GonzaloA/fake_news', 'text', 'label', 1, 0),
        ('mohammadjavadpirhadi/fake-news-detection-dataset-english', 'text', 'label', 0, 1),
        ('nanyy1025/covid_fake_news', 'tweet', 'label', 'real', 'fake'),
        ('ikekobby/40-percent-cleaned-preprocessed-fake-real-news', 'article', 'label', 1, 0),
        ('roupenminassian/twitter-misinformation', 'text', 'label', 0, 1),
        ('Intel/misinformation-guard', 'text', 'label', 3, 0),
        ('argilla/news-fakenews', 'text', 'prediction', 'real', 'fake'),
        ('pushpdeep/fake_news_combined', 'text', 'label', 1, 0),
        ('pushpdeep/fake_news_test', 'text', 'label', 1, 0),
        ('andyP/fake_news_en_opensources', 'content', 'type', None, 'fake'),
        ('AlexanderHolmes0/true-fake-news', 'text', 'label', 0, 1),
        ('Hasib18/fake-news-dataset', 'text', 'label', 1, 0),
        ('nixbel/fakenews_train', 'Content', 'Label', 'Credible', 'Suspicious'),
        ('lusamaki/Fake_News_Detection_System_29.5k', 'tweet', 'label', 'fact', 'fake')
    ]

    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    #  create file handler
    fh = logging.FileHandler(args.log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # set a unified log format
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the two handlers
    logger.addHandler(fh)
    logger.addHandler(ch)

    for info in meta_info:
        cached_path = f'train_data/cache/{info[0].replace("/", "_")}'
        logger.info("*" * 20)
        cached_path = f'train_data/cache/{info[0].replace('/', '_')}'
        loaded_data = load_md_datasets(info[0], cached_path, info[1], info[2], info[3], info[4])
        all_data = check_overlap_and_merge(all_data, loaded_data, args.overlap_char)

    write_jsonl(all_data, args.output_path.format(num=len(all_data)))
