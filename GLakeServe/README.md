# GlakeServe: Flexible Virtual Tensor Management for Efficient LLM Serving

## Overview

**GlakeServe** is dedicated to Large Language Model (LLM) inference, specifically designed to address the escalating demands for optimized throughput, reduced latency, and cost-effectiveness. Targeting the core challenge of managing Key-Value (KV) caches and maintaining compute as well as memory flexibility, GlakeServe facilitates LLM inference by leveraging GPU virtual memory management (VMM) principles.

Tips: We are trying to merge all functionalities from different versions of GlakeServe, which was originally designed and developed in v0.2.1, into one repo and simplify its building process. Features like prefix-caching, cuda-graph and flexible memory usage will be migrated into this repo soon. Currently, this repo only supports single QA serving workload on llama architecture. 

## Key Features

- **Decoupled Computation & Memory Management**: 
This repo leverages VMM technique to construct an efficient abstraction called "vTensor" to manage the allocation of KV Cache dynamically. With the continuous virtual memory address, we can decouple the implementation of computation kernels from LLM memory management. 

- **Dynamic Extensibility**: GlakeServe supports evolving LLM requirements and varying workload patterns without compromising performance.

- **Heterogeneous CPU-GPU Strategy**: We implement support for asynchronous KV Cache memory orchestration in LLM decoding stage.

- **Fragmentation-Free Memory**: Guarantees efficient memory usage without the typical pitfalls of memory fragmentation. 


## Get Started

- 📦 **Installation**:

```
pip install -r requirements.txt # install dependencies
cd vmm_allocator
g++ -shared -std=c++14 -fPIC `python3 -m pybind11 --includes` vmm_allocator.cpp -o vmmAllocator`python3-config --extension-suffix` -I python -I/usr/local/cuda/include -L/usr/lib64/ -l cuda -I. # build vmmAllocator
mv vmmAllocator.cpython-310-x86_64-linux-gnu.so /usr/lib64/
cd ../
git clone https://github.com/intelligent-machine-learning/glake-flash-attn.git
cd glake-flash-attn
python setup.py install
cd ../
python setup.py install # build GlakeServe
```

- 🏃‍♂️ **Benchmarking Guide**: [Benchmarking](./benchmarks/README.md)


## Acknowledgement

GlakeServe is developed with the help of diverse open-source projects like [vLLM](https://github.com/vllm-project/vllm/tree/main), [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/main) and [SGLang](https://github.com/sgl-project/sglang). Valuable advice from the community also contributes to this repo and is deeply appreciated by us.

## Citation

If you find GlakeServe helpful to your research, please cite our [paper](https://arxiv.org/abs/2407.15309):

```bibtex
@misc{xu2024vtensorflexiblevirtualtensor,
title={vTensor: Flexible Virtual Tensor Management for Efficient LLM Serving}, 
author={Jiale Xu and Rui Zhang and Cong Guo and Weiming Hu and Zihan Liu and Feiyang Wu and Yu Feng and Shixuan Sun and Changxu Shao and Yuhong Guo and Junping Zhao and Ke Zhang and Minyi Guo and Jingwen Leng},
year={2024},
eprint={2407.15309},
archivePrefix={arXiv},
primaryClass={cs.DC},
url={https://arxiv.org/abs/2407.15309}, 
}
```

## License

GlakeServe is released under the [Apache 2.0 License](LICENSE).