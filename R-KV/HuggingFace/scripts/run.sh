export CUDA_VISIBLE_DEVICES=0

python3 ./run_math.py \
--dataset_path ./data/aime24.jsonl \
--save_path ./outputs/output.jsonl \
--model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
--max_length 16384 \
--eval_batch_size 1 \
--method rkv \
--kv_budget 128
# --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
# --model_path alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B \
