#!/bin/bash

set -e

# ========================================
# Configuration
# ========================================
CONDA_ENV="codesim"
MODEL1_PATH="/scr1/yixuand/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct"
MODEL2_PATH="/scr1/yixuand/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct"
MODEL1_PORT=8000
MODEL2_PORT=8001
MAX_MODEL_LEN=16384
MODEL1_GPU_MEM=0.2
MODEL2_GPU_MEM=0.4


MODEL1_GPU="2"
MODEL2_GPU="1"

# CodeSIM parameters
DATASET="HumanEval"
STRATEGY="CodeSIM"
LANGUAGE="Python3"
PASS_AT_K=1
TEMPERATURE=0.0
TOP_P=0.95
VERBOSE=2
MODEL_PROVIDER="OpenAI"
CONT="yes"
RESULT_LOG="partial"
STORE_LOG_IN_FILE="yes"

# ========================================
# Setup
# ========================================
eval "$(conda shell.bash hook)"
cd /scr1/yixuand/CodeGenerator

# Clean up existing services
echo "Cleaning up existing services..."
tmux kill-session -t model1 2>/dev/null || true
tmux kill-session -t model2 2>/dev/null || true
tmux kill-session -t codesim 2>/dev/null || true
fuser -k $MODEL1_PORT/tcp 2>/dev/null || true
fuser -k $MODEL2_PORT/tcp 2>/dev/null || true
sleep 2

# ========================================
# Start Models
# ========================================
echo "Starting Model 1 (1.5B) on GPU $MODEL1_GPU, port $MODEL1_PORT..."
tmux new-session -d -s model1 "eval \"\$(conda shell.bash hook)\" && conda activate $CONDA_ENV && \
CUDA_VISIBLE_DEVICES=$MODEL1_GPU vllm serve $MODEL1_PATH \
    --host 0.0.0.0 \
    --port $MODEL1_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $MODEL1_GPU_MEM"

sleep 3

echo "Starting Model 2 (7B) on GPU $MODEL2_GPU, port $MODEL2_PORT..."
tmux new-session -d -s model2 "eval \"\$(conda shell.bash hook)\" && conda activate $CONDA_ENV && \
CUDA_VISIBLE_DEVICES=$MODEL2_GPU vllm serve $MODEL2_PATH \
    --host 0.0.0.0 \
    --port $MODEL2_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $MODEL2_GPU_MEM"

# ========================================
# Wait for Models
# ========================================
echo "Waiting for models to be ready..."

# Wait for Model 1
for i in {1..60}; do
    if curl -s http://localhost:$MODEL1_PORT/v1/models >/dev/null 2>&1; then
        echo "Model 1 ready!"
        break
    fi
    [ $i -eq 60 ] && echo "Model 1 timeout!" && exit 1
    sleep 5
done

# Wait for Model 2
for i in {1..60}; do
    if curl -s http://localhost:$MODEL2_PORT/v1/models >/dev/null 2>&1; then
        echo "Model 2 ready!"
        break
    fi
    [ $i -eq 60 ] && echo "Model 2 timeout!" && exit 1
    sleep 5
done

echo ""
echo "Both models ready! Starting CodeSIM in tmux..."
echo ""

# ========================================
# Run CodeSIM in tmux
# ========================================
tmux new-session -d -s codesim "eval \"\$(conda shell.bash hook)\" && conda activate $CONDA_ENV && \
cd /scr1/yixuand/CodeGenerator && \
python -u src/main.py \
    --dataset $DATASET \
    --strategy $STRATEGY \
    --language $LANGUAGE \
    --pass_at_k $PASS_AT_K \
    --model1 \"$MODEL1_PATH\" \
    --model2 \"$MODEL2_PATH\" \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --verbose $VERBOSE \
    --model_provider $MODEL_PROVIDER \
    --cont $CONT \
    --result_log $RESULT_LOG \
    --store_log_in_file $STORE_LOG_IN_FILE; \
echo 'CodeSIM finished. Press Ctrl+C to close or Ctrl+B D to detach.'; \
bash"

echo "All services started in tmux!"
echo ""
echo "Tmux sessions:"
echo "  model1:  tmux attach -t model1   (Model 1 - 1.5B)"
echo "  model2:  tmux attach -t model2   (Model 2 - 7B)"
echo "  codesim: tmux attach -t codesim  (CodeSIM experiment)"
echo ""
echo "To stop all: ./stop_models.sh"
