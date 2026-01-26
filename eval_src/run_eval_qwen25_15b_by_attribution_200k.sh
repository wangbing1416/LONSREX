#!/bin/bash

MODEL="${1:-YOURMODELPATH}"  # pass model path as first argument or set MODEL_PATH env variable
if [ -z "$MODEL" ]; then
  echo "MODEL is not set. Pass as first argument or set MODEL_PATH."
  exit 1
fi

bash train_data/start_sglang_qwen2.sh "$MODEL" &   # start server in background
python gpu_stress.py 300     # wait for server readiness
python eval_src/eval_mdbench_vanilla.py \
    --dataset gossipcop \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_vanilla \
    --num_processes 128

python eval_src/eval_mdbench_vanilla.py \
    --dataset politifact \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_vanilla \
    --num_processes 128

python eval_src/eval_mdbench_vanilla.py \
    --dataset multidis \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_vanilla \
    --num_processes 128

python eval_src/eval_mdbench_vanilla.py \
    --dataset eudisinfo \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_vanilla \
    --num_processes 128

python eval_src/eval_mdbench_cot.py \
    --dataset gossipcop \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_cot \
    --num_processes 128

python eval_src/eval_mdbench_cot.py \
    --dataset politifact \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_cot \
    --num_processes 128

python eval_src/eval_mdbench_cot.py \
    --dataset multidis \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_cot \
    --num_processes 128

python eval_src/eval_mdbench_cot.py \
    --dataset eudisinfo \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_cot \
    --num_processes 128

python eval_src/eval_mdbench_arg.py \
    --dataset gossipcop \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_arg \
    --num_processes 128

python eval_src/eval_mdbench_arg.py \
    --dataset politifact \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_arg \
    --num_processes 128

python eval_src/eval_mdbench_arg.py \
    --dataset multidis \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_arg \
    --num_processes 128

python eval_src/eval_mdbench_arg.py \
    --dataset eudisinfo \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_arg \
    --num_processes 128

python eval_src/eval_mdbench_genfend.py \
    --dataset gossipcop \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_genfend \
    --num_processes 128

python eval_src/eval_mdbench_genfend.py \
    --dataset politifact \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_genfend \
    --num_processes 128

python eval_src/eval_mdbench_genfend.py \
    --dataset multidis \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_genfend \
    --num_processes 128

python eval_src/eval_mdbench_genfend.py \
    --dataset eudisinfo \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_genfend \
    --num_processes 128

python eval_src/eval_mdbench_pcot.py \
    --dataset gossipcop \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_pcot \
    --num_processes 128

python eval_src/eval_mdbench_pcot.py \
    --dataset politifact \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_pcot \
    --num_processes 128

python eval_src/eval_mdbench_pcot.py \
    --dataset multidis \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_pcot \
    --num_processes 128

python eval_src/eval_mdbench_pcot.py \
    --dataset eudisinfo \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_pcot \
    --num_processes 128

python eval_src/eval_mdbench_kc_onestep.py \
    --dataset gossipcop \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_kc_onestep \
    --num_processes 128

python eval_src/eval_mdbench_kc_onestep.py \
    --dataset politifact \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_kc_onestep \
    --num_processes 128

python eval_src/eval_mdbench_kc_onestep.py \
    --dataset multidis \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_kc_onestep \
    --num_processes 128

python eval_src/eval_mdbench_kc_onestep.py \
    --dataset eudisinfo \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_kc_onestep \
    --num_processes 128

python eval_src/eval_mdbench_kc_twostep.py \
    --dataset gossipcop \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_kc_twostep \
    --num_processes 128

python eval_src/eval_mdbench_kc_twostep.py \
    --dataset politifact \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_kc_twostep \
    --num_processes 128

python eval_src/eval_mdbench_kc_twostep.py \
    --dataset multidis \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_kc_twostep \
    --num_processes 128

python eval_src/eval_mdbench_kc_twostep.py \
    --dataset eudisinfo \
    --model "$MODEL" \
    --output_path eval_qwen25_15b_xmdcot_attribution_200k_kc_twostep \
    --num_processes 128
