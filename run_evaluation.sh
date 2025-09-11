#!/bin/bash
cd /home/datagen/healthcare_reasoning/fine-tuning
# Exit on any error
set -e

# --- Configuration ---
MODEL_PATH="/data0/model/Qwen2.5-7B-Instruct-healthcare-cot"
# MODEL_PATH="/data0/model/Qwen2.5-7B-Instruct"
API_PORT=8000
API_URL="http://localhost:${API_PORT}/v1/chat/completions"
HEALTH_CHECK_URL="http://localhost:${API_PORT}/health"
# Use all 8 available GPUs for tensor parallelism
GPU_DEVICE="0,1,2,3"
NUM_GPUS=4
LOG_FILE="vllm_server.log"

# --- Cleanup ---
# Initialize PID to an invalid value
API_SERVER_PID=0

# Cleanup function to be called on script exit
cleanup() {
    echo "Performing cleanup..."
    # Check if the process is still running
    if [ $API_SERVER_PID -ne 0 ] && kill -0 $API_SERVER_PID > /dev/null 2>&1; then
        echo "Shutting down API server (PID: $API_SERVER_PID)..."
        kill $API_SERVER_PID
        wait $API_SERVER_PID 2>/dev/null
    else
        echo "API server not running or already shut down."
    fi
    echo "Cleanup complete."
}

# Set the trap to call the cleanup function on EXIT, INTERRUPT, or TERM signal
trap cleanup EXIT INT TERM

# --- Main Script ---

echo "Starting vLLM OpenAI-compatible API server on ${NUM_GPUS} GPUs... Logs will be saved to ${LOG_FILE}"
# Clear previous log file
> "$LOG_FILE"
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size $NUM_GPUS \
    --host 0.0.0.0 \
    --port $API_PORT \
    --trust-remote-code > "$LOG_FILE" 2>&1 &

# Get the process ID of the background server
API_SERVER_PID=$!

echo "API server starting with PID: $API_SERVER_PID. Waiting for it to become available..."

# Wait for the API server to be ready
RETRY_COUNT=0
MAX_RETRIES=100
# The chat endpoint (/v1/chat/completions) doesn't support HEAD requests for health checks.
# Instead, we poll the standard /health endpoint which is designed for this purpose.
until $(curl --output /dev/null --silent --fail $HEALTH_CHECK_URL); do
    if [ ${RETRY_COUNT} -ge ${MAX_RETRIES} ]; then
        echo "API server failed to start in time. Aborting."
        exit 1 # This will trigger the cleanup trap
    fi
    printf '.'
    RETRY_COUNT=$(($RETRY_COUNT+1))
    sleep 2
done

echo "\nAPI server is up and running!"

echo "Server is up. Running evaluation script..."
# python evaluate_finetuned_model.py \
python evaluate_testallinfo_model.py \
    --model_path "$MODEL_PATH" \
    --api_url $API_URL \
    --output_csv "predictions_vllm_$(basename $MODEL_PATH).csv" \
    --num_workers 64

echo "Evaluation finished successfully." 