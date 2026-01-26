# Are Rationales Necessary and Sufficient? Tuning LLMs for Explainable Misinformation Detection

<p align="center">
<img src="assert/logo.jpg" alt="logo" width="450">
</p>

Raw data (316k instances), filtered data (231k instances), and the model checkpoint will be released upon paper acceptance.

This repository contains the source code and analysis scripts for the paper *"Are Rationales Necessary and Sufficient? Tuning LLMs for Explainable Misinformation Detection"*.

## 🎨 Highlights

- We empirically analyze **unnecessary and insufficient rationales** in SFT datasets for explainable misinformation detection;
- A **data construction pipeline LONSREX** is proposed to locate necessary and sufficient rationales;

## Environment Setup

We primarily use [SGLang](https://github.com/sgl-project/sglang) for model deployment, data generation, and label perplexity computation,
and [360-LLaMA-Factory](https://github.com/Qihoo360/360-LLaMA-Factory) for model training. We recommend using the latest environment or directly leveraging their official Docker images for experimentation.

---

## Data Preparation

### SFT Datasets (Article Collection)

The SFT datasets are constructed from several public sources. A representative list is shown below:

```
[
    ('GonzaloA/fake_news', 'text', 'label', 1, 0),
    ('mohammadjavadpirhadi/fake-news-detection-dataset-english', 'text', 'label', 0, 1),
    ('nanyy1025/covid_fake_news', 'tweet', 'label', 'real', 'fake'),
    ('ikekobby/40-percent-cleaned-preprocessed-fake-real-news', 'article', 'label', 1, 0),
    ('roupenminassian/twitter-misinformation', 'text', 'label', 0, 1),
    ('Intel/misinformation-guard', 'text', 'label', 3, 0),
    ('argilla/news-fakenews', 'text', 'prediction', 'real', 'fake'),
    ('pushpdeep/fake_news_test', 'text', 'label', 1, 0),
    ('andyP/fake_news_en_opensources', 'content', 'type', None, 'fake'),
    ('AlexanderHolmes0/true-fake-news', 'text', 'label', 0, 1),
    ('Hasib18/fake-news-dataset', 'text', 'label', 1, 0),
    ('nixbel/fakenews_train', 'Content', 'Label', 'Credible', 'Suspicious'),
]
```
We provide a data preparation script `train_data/data_collection.py` to download and preprocess these datasets.


### SFT Datasets (Rationale Generation)

**Step 1**: You can generate rationales for the above articles under the SGLang environment by running the following script:

```bash
# an example for Qwen3-30B-A3B
bash train_data/start_sglang_qwen3_a3b_instruct.sh  # deploy the model in SGLang
python train_data/data_gen_qwen3_a3b.py  # generate rationales
```

**Step 2**: Merge the generated rationales with the original articles by running:

```bash
# an example for Qwen2.5-1.5B
python train_data/merge_qwen2_data.py
```

**Step 3**: Obtain single-step attributions for each SFT instance by running:

```bash
# an example for Qwen2.5-1.5B
bash train_data/start_sglang_qwen2.sh  # deploy the model in SGLang
python train_data/output_logits.py  # obtain single-step attributions
```

**Step 4**: Finally, you can filter the SFT datasets using our proposed LONSREX method by running:

```bash
# an example of our ablation version
python train_data/filter_attribution_data.py
```

### Evaluation Datasets

We evaluate our models on four benchmark datasets: GossipCop++, PolitiFact++, MultiDis, EuDisinfo. We process these datasets using the scripts `process_data.py` in the `md_data` folder.

The processed datasets will be saved in the `md_data/` folder.
The folder structure is as follows:
```folder
md_data/
├── processed_gossipcop.jsonl -> {id, text, title, label}
├── processed_politifact.jsonl -> {id, text, title, description, label}
├── processed_eudisinfo.jsonl -> {id, text, label}
└── processed_multidis.jsonl -> {id, text, label}
```

---
## Model Training

After activating the [360-LLaMA-Factory](https://github.com/Qihoo360/360-LLaMA-Factory) environment, you can train the models using the following command:

```bash
cd yourLlamaFactoryEnvPath
```
Note: update `EXPERIMENT_CONFIG` and other variables to match your environment before running.
```bash
NNODES="${WORLD_SIZE:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
RANK="${RANK:-0}"
PROCESS_PER_NODE="${PROCESS_PER_NODE:-8}"
NGPUS="${TQ_GPU_NUM:-8}"

EXPERIMENT_CONFIG="/rootpath/sft_task/src/full_sft_qwen25_15b_by_attribution_200k.yaml"

torchrun --nproc_per_node $NGPUS --nnodes $NNODES \
    --rdzv_id=3649 --rdzv_backend=c10d  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    src/llamafactory/launcher.py $EXPERIMENT_CONFIG
```

## Evaluation

Evaluation scripts are provided in the `eval_src/` folder. To run the full evaluation pipeline for a trained checkpoint, use:

```bash
bash eval_src/run_eval_qwen25_15b_by_attribution_200k.sh
```

Individual evaluation utilities (examples):

```bash
python eval_src/eval_mdbench_vanilla.py  # direct evaluation without rationales
python eval_src/eval_mdbench_cot.py      # evaluation with chain-of-thought rationales
python eval_src/eval_mdbench_arg.py      # evaluation with ARG rationales (text/commonsense)
python eval_src/eval_mdbench_genfend.py  # evaluation with GenFend rationales
python eval_src/eval_mdbench_pcot.py     # evaluation with PCoT rationales
python eval_src/eval_mdbench_kc_onestep.py  # evaluation with DMR one-step knowledge rationales
python eval_src/eval_mdbench_kc_twostep.py  # evaluation with DMR two-step knowledge rationales
```

Evaluation outputs are saved to `eval_results/`. Example outputs and metrics are provided for reference.

---

## Preliminary analysis

Preliminary analysis scripts used to reproduce figures and statistics are included. Example:

```bash
python preliminary_analysis.py
```

---

## Tips

If you have questions or find issues, please open an issue in this repository. During anonymization some file paths and identifying details may have been redacted — please verify and update paths and configuration before running the code.

---

## Citation

If you find our paper, data, or code useful, we would greatly appreciate it if you could cite the following article.

```bibtex
Citation details will be provided once the paper is accepted.
```