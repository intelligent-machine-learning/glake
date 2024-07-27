 # Downloading the dataset

You can download the dataset by running:
```
# in this directory

wget https://huggingface.co/datasets/Infinigence/LVEval/resolve/main/hotpotwikiqa_mixup.zip

wget https://huggingface.co/datasets/Infinigence/LVEval/resolve/main/loogle_SD_mixup.zip

unzip hotpotwikiqa_mixup.zip
unzip loogle_SD_mixup.zip

python jsonl.py

```
# Benchmarking (only llama architecture is supported by now!)

```

# launch the api server
python -m vllm.entrypoints.openai.api_server --model <your_model_path> --swap-space 0 --disable-log-requests --port 21008 --cache-seqs 64 --cache-tokens 4  
--tensor-parallel-size 1 

# launch the client process
python benchmark_serving.py         --backend vllm         --model <your_model_path>         --num-prompts 400 --request-rate 0.4 --port 21008

```


