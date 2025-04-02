export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
./scripts/train/run_deepscaler_1.5b_8k.sh --model $MODEL_PATH