#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"

MODEL="Qwen/Qwen3-0.6B"
HOST="0.0.0.0"
PORT="1212"
MAX_MODEL_LEN="" 
TENSOR_PARALLEL=1

# 构建基础命令
CMD="vllm serve ${MODEL} \
    --host ${HOST} \
    --port ${PORT} \
    --tensor-parallel-size ${TENSOR_PARALLEL}\
    "

if [ -n "${MAX_MODEL_LEN}" ]; then 
    CMD+=" --max-model-len ${MAX_MODEL_LEN}"
fi

echo "${CMD}"
eval "${CMD}"