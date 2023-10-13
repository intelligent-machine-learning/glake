#pragma once

#include <cuda.h>

#ifdef __cplusplus
extern "C" {

namespace glake {
#endif

int SwapOut(void* host_addr, void* dev_addr, size_t bytes,
            const CUstream& stream);
int SwapIn(void* dev_addr, void* host_addr, size_t bytes,
           const CUstream& stream);
int H2DMultiPath(void* dev_addr, void* host_addr, size_t bytes,
                 CUstream& stream);
int D2HMultiPath(void* host_addr, void* dev_addr, size_t bytes,
                 CUstream& stream);

CUresult CUDAAPI fetch(void* dst, void* src, size_t ByteCount,
                       CUstream hStream);
CUresult CUDAAPI evict(void* src, size_t ByteCount, CUstream hStream);

#ifdef __cplusplus
}  // namespace glake

}  // entern "C"
#endif
