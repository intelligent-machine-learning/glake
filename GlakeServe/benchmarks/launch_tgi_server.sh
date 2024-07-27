#!/bin/bash

PORT=8000
MODEL=$1
TOKENS=16384

pouch run --gpus all --shm-size 1g -p $PORT:80 \
           -v $PWD/home/george.zr:/home/george.zr \
           ghcr.io/huggingface/text-generation-inference:1.4.0 \
           --model-id $MODEL \
           --sharded false  \
           --max-input-length 8000 \
           --max-total-tokens 6000 \
           --max-best-of 1 \
           --max-concurrent-requests 256 \
           --max-batch-total-tokens $TOKEN$
