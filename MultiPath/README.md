GLake Multi-path speeds up CPU-GPU IO transmission by exploiting NvLink and multiple PCIe paths on the same node.

Build:
```bash
$ cd src
$ wget https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.4.tar.gz
$ tar xvzf v2.4.tar.gz
$ cd gdrcopy-2.4/src; make
$ cd -; make
```

Usage:
As MultiPath hooks CUDA driver API (e.g., cuMemcpyHtoDAsync), it is not needed to change code. Rather, use environment variable GLAKE_MULTI_PATH=1 to turn on Multi-path to speed up your CUDA applications.

Run benchmake:
```bash
$ cd ../test
$ make bench
```

Result:
We got bandwidth 83.1 GB/s for HostToDevice and 71.2 GB/s for DeviceToHost on a node of 8 * A100(80GB) connected with NvLink.
