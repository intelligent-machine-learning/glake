## Motivation
CPU-GPU is connected with PCIe. Due to the limited PCIe bandwidth (~24GB/s for gen4), CPU-GPU IO is a common bottleneck for GPU applications.

## What is Multi-path?
GLake Multi-path speeds up CPU-GPU IO by exploiting NvLink and multiple PCIe paths on the same node.

## Build
Make sure CUDA (version>=11.6) is installed in `/usr/local/cuda`.
```bash
$ cd src
$ wget https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.4.tar.gz
$ tar xvzf v2.4.tar.gz
$ cd gdrcopy-2.4/src; make
$ cd -; make
```

## Usage
As Multi-path hooks CUDA driver API (e.g., `cuMemcpyHtoDAsync`), it is not needed to modify your code. Rather, use environment variable `GLAKE_MULTI_PATH=1` to turn on Multi-path.

## Run benchmark
```bash
$ cd test
# Note sudo permission is required.
$ make bench
```

## Preliminary Result
We got bandwidth 83.1 GB/s for HostToDevice and 71.2 GB/s for DeviceToHost on a node of 8 * A100(80GB), where GPUs are connected with NvLink and four PCIe paths are located between CPU-GPU.
