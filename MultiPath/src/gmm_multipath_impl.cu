#include "gmm_cuda_common.h"
#include "gmm_mp.h"

// launch cp kernel on given stream to perf copy
int gmm_launch_cp_kernel(char *tgt_addr, char *src_addr, size_t bytes,
                         CUstream stream, CUevent &pre_evt, CUevent &post_evt,
                         bool sync) {
  int ret = 0;

  uint64_t num_thread_blocks;
  uint64_t num_elements_in_thread_block =
      NUM_LOOP_UNROLL * NUM_THREADS_IN_BLOCK;
  uint64_t num_bytes_in_thread_block =
      num_elements_in_thread_block * sizeof(ulong2);
  int cur_dev = 0;
  CHECK_CUDA(cudaGetDevice(&cur_dev));

  if (bytes % num_bytes_in_thread_block) {
    LOGGER(ERROR, "Data size %ld should be multiple of %lu\n", bytes,
           num_bytes_in_thread_block);
    return -1;
  }

  num_thread_blocks = bytes / num_bytes_in_thread_block;
  cudaEvent_t start, stop;

  if (pre_evt) {
    CHECK_DRV(cuStreamWaitEvent(stream, pre_evt, CU_EVENT_WAIT_DEFAULT));
    LOGGER(DEBUG, "log:%d cur-dev:%d stramWait pre_evt done", gmm_log_level,
           cur_dev);
  }

  if (sync) {
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, stream));
  }

  // launch kernel for copy
  // TODO: fine tune kernel size
  SMCopyKernel<<<num_thread_blocks, NUM_THREADS_IN_BLOCK, 0, stream>>>(
      reinterpret_cast<ulong2 *>(tgt_addr),
      reinterpret_cast<ulong2 *>(src_addr));
  if (post_evt) {
    CHECK_DRV(cuEventRecord(post_evt, stream));
  }

  if (sync) {
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float time_in_ms;
    CHECK_CUDA(cudaEventElapsedTime(&time_in_ms, start, stop));
    LOGGER(INFO, "cur_dev:%d bytes:%ld BW:%f GB/sec", cur_dev, bytes,
           bytes / (1000000.0 * time_in_ms));
  }

  return ret;
}

// start DMA cp on given stream
int gmm_launch_cp_DMA(char *tgt_addr, char *src_addr, char *tmp_buf,
                      size_t bytes, size_t buf_bytes, cudaStream_t stream,
                      CUevent &pre_evt, CUevent &post_evt, bool sync = false) {
  int ret = 0;

  if (sync) CHECK_CUDA(cudaStreamSynchronize(stream));

  size_t offset = 0;
  size_t left = bytes;
  size_t to_cp_bytes = buf_bytes;

  if (pre_evt) {
    CHECK_DRV(cuStreamWaitEvent(stream, pre_evt, CU_EVENT_WAIT_DEFAULT));
  }

  while (left > 0) {
    to_cp_bytes = (left >= buf_bytes) ? buf_bytes : left;
    CHECK_CUDA(cudaMemcpyAsync(tmp_buf, src_addr + offset, to_cp_bytes,
                               cudaMemcpyDefault, stream));
    CHECK_CUDA(cudaMemcpyAsync(tgt_addr + offset, tmp_buf, to_cp_bytes,
                               cudaMemcpyDefault, stream));
    offset += to_cp_bytes;
    left -= to_cp_bytes;
  }
  CHECK_DRV(cuEventRecord(post_evt, stream));

  if (sync) {
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }

  return ret;
}

// no tmp buf, just issue memcpy
int gmm_DMA_direct(char *tgt_addr, char *src_addr, size_t bytes,
                   cudaStream_t stream, CUevent &pre_evt, CUevent &post_evt,
                   bool sync = false) {
  int ret = 0;

  cudaEvent_t start, stop;

  if (sync) {
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, stream));
  }

  if (pre_evt) {
    CHECK_DRV(cuStreamWaitEvent(stream, pre_evt, CU_EVENT_WAIT_DEFAULT));
  }

  CHECK_CUDA(cudaMemcpyAsync(tgt_addr, src_addr, bytes,
                             cudaMemcpyDeviceToDevice, stream));

  CHECK_DRV(cuEventRecord(post_evt, stream));

  if (sync) {
    // CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float time_in_ms;
    CHECK_CUDA(cudaEventElapsedTime(&time_in_ms, start, stop));
    LOGGER(INFO, "bytes:%ld BW:%f GB/sec", bytes,
           bytes / (1000000.0 * time_in_ms));
  }

  return ret;
}

// start DMA cp on given stream
int gmm_pipeline_DMA(char *tgt_addr, char *src_addr, size_t bytes,
                     char *tmp_buf1, char *tmp_buf2, size_t buf_bytes,
                     cudaStream_t pipe_stream1, cudaStream_t pipe_stream2,
                     cudaEvent_t &pipe_evt1, cudaEvent_t &pipe_evt2,
                     CUevent &pre_evt, CUevent &post_evt, bool sync = false) {
  int ret = 0;

  if (sync) {
    CHECK_CUDA(cudaStreamSynchronize(pipe_stream1));
    CHECK_CUDA(cudaStreamSynchronize(pipe_stream2));
  }

  size_t offset = 0;
  size_t left = bytes;
  size_t to_cp_bytes = buf_bytes;

  if (pre_evt) {
    CHECK_DRV(cuStreamWaitEvent(pipe_stream1, pre_evt, CU_EVENT_WAIT_DEFAULT));
    CHECK_DRV(cuStreamWaitEvent(pipe_stream2, pre_evt, CU_EVENT_WAIT_DEFAULT));
  }
  // s1: pcie_cp   nvlink
  // s2            pcie_cp  nvlink
  bool s1_last = true;
  while (left > 0) {
    to_cp_bytes = (left >= buf_bytes) ? buf_bytes : left;

    CHECK_CUDA(cudaStreamWaitEvent(pipe_stream1, pipe_evt2));
    CHECK_CUDA(cudaMemcpyAsync(tmp_buf1, src_addr + offset, to_cp_bytes,
                               cudaMemcpyDefault, pipe_stream1));
    CHECK_CUDA(cudaEventRecord(pipe_evt1, pipe_stream1));
    CHECK_CUDA(cudaMemcpyAsync(tgt_addr + offset, tmp_buf1, to_cp_bytes,
                               cudaMemcpyDefault, pipe_stream1));

    left -= to_cp_bytes;
    offset += to_cp_bytes;
    s1_last = true;

    if (left > 0) {
      to_cp_bytes = (left >= buf_bytes) ? buf_bytes : left;

      CHECK_CUDA(cudaStreamWaitEvent(pipe_stream2, pipe_evt1));
      CHECK_CUDA(cudaMemcpyAsync(tmp_buf2, src_addr + offset, to_cp_bytes,
                                 cudaMemcpyDefault, pipe_stream2));
      CHECK_CUDA(cudaEventRecord(pipe_evt2, pipe_stream2));
      CHECK_CUDA(cudaMemcpyAsync(tgt_addr + offset, tmp_buf2, to_cp_bytes,
                                 cudaMemcpyDefault, pipe_stream2));

      left -= to_cp_bytes;
      offset += to_cp_bytes;
      s1_last = false;
    }
  }
  if (s1_last) {
    CHECK_DRV(cuEventRecord(post_evt, pipe_stream1));
  } else {
    CHECK_DRV(cuEventRecord(post_evt, pipe_stream2));
  }

  if (sync) {
    CHECK_CUDA(cudaStreamSynchronize(pipe_stream1));
    CHECK_CUDA(cudaStreamSynchronize(pipe_stream2));
  }

  return ret;
}

// verify data correctness btw baseline and to_verify
// return:0 ok; others: failed
int gmm_verify_data(void *const &to_verify, void *const &truth, size_t bytes,
                    bool truth_at_cpu = true, bool to_verify_at_gpu = true) {
  int ret = -1;

  if (truth_at_cpu && to_verify_at_gpu) {
    char *check_buf = (char *)malloc(bytes);
    ASSERT(check_buf, "Failed on malloc");
    CHECK_CUDA(cudaMemcpy(check_buf, to_verify, bytes, cudaMemcpyDeviceToHost));
    ret = memcmp(truth, check_buf, bytes);
    free(check_buf);
  } else if (truth_at_cpu && !to_verify_at_gpu) {
    ret = memcmp(to_verify, truth, bytes);
  }

  LOGGER(INFO, "memory check with ground-truth: %s", ret ? "failed" : "ok");

  return ret;
}

// Fetch a ulong2 from source memory and write to register
// 1) NCCL:
// https://github.com/NVIDIA/nccl/blob/7e515921295adaab72adf56ea71a0fafb0ecb5f3/src/collectives/device/common_kernel.h#L483
// 2) RCCL:
// https://github.com/ROCmSoftwarePlatform/rccl/blob/5c8380ff5b5925cae4bce00b1879a5f930226e8d/src/collectives/device/common_kernel.h#L268
static inline __device__ void FetchULong2(ulong2 &v, const ulong2 *p) {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
  v.x = p->x;
  v.y = p->y;
#else
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];"
               : "=l"(v.x), "=l"(v.y)
               : "l"(p)
               : "memory");
#endif
}

// Store a ulong2 from register and write to target memory
// 1) NCCL:
// https://github.com/NVIDIA/nccl/blob/7e515921295adaab72adf56ea71a0fafb0ecb5f3/src/collectives/device/common_kernel.h#L486
// 2) RCCL:
// https://github.com/ROCmSoftwarePlatform/rccl/blob/5c8380ff5b5925cae4bce00b1879a5f930226e8d/src/collectives/device/common_kernel.h#L276
static inline __device__ void StoreULong2(ulong2 *p, ulong2 &v) {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
  p->x = v.x;
  p->y = v.y;
#else
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" ::"l"(p), "l"(v.x),
               "l"(v.y)
               : "memory");
#endif
}

// Fetch data from source memory into register first, and then write them to
// target memory Stride set to thread block size to best utilize cache src
// -fetch()-> local-var,  local-var ->store() -> tgt
__global__ void SMCopyKernel(ulong2 *tgt, const ulong2 *src) {
  uint64_t index = blockIdx.x * blockDim.x * NUM_LOOP_UNROLL + threadIdx.x;
  ulong2 val[NUM_LOOP_UNROLL];
#pragma unroll
  for (uint64_t i = 0; i < NUM_LOOP_UNROLL; i++)
    FetchULong2(val[i], src + index + i * blockDim.x);
#pragma unroll
  for (uint64_t i = 0; i < NUM_LOOP_UNROLL; i++)
    StoreULong2(tgt + index + i * blockDim.x, val[i]);
}

/*
static inline int set_gpu(int gpu_id) {
  CHECK_CUDA(cudaSetDevice(gpu_id));
  return 0;
}

void * htod_direct(void *args)
{
  gmm_mp_arg_t *p = (gmm_mp_arg_t *)args;
  set_gpu(p->gpu_id);
  LOGGER(INFO, "Multi-path[%d] thr GPU:%d via-DMA:%d src:%p dst:%p bytes:%lu\n",
p->path_id, p->gpu_id, p->is_dma, p->src_addr, p->dst_addr, p->cp_bytes);

  CHECK_CUDA(cudaEventRecord(p->start_event, p->stream1));
  for (int i = 0; i < p->iter; i++) {
    CHECK_CUDA(cudaMemcpyAsync(p->dst_addr, p->src_addr, p->cp_bytes,
cudaMemcpyDefault, p->stream1));
  }
  CHECK_CUDA(cudaEventRecord(p->end_event, p->stream1));

  CHECK_CUDA(cudaEventSynchronize(p->end_event));
  float time_in_ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&time_in_ms, p->start_event, p->end_event));

  LOGGER(INFO, "Multi-path[%d] thr GPU:%d BW:%.2f GB/Sec\n", p->path_id,
p->gpu_id, p->cp_bytes * p->iter / time_in_ms / 1e6);

  return nullptr;
}

void * dtoh_direct(void *args)
{
  gmm_mp_arg_t *p = (gmm_mp_arg_t *)args;
  set_gpu(p->gpu_id);
  LOGGER(INFO, "Multi-path[%d] thr GPU:%d via-DMA:%d src:%p dst:%p bytes:%lu\n",
p->path_id, p->gpu_id, p->is_dma, p->src_addr, p->dst_addr, p->cp_bytes);

  CHECK_CUDA(cudaEventRecord(p->start_event, p->stream1));
  for (int i = 0; i < p->iter; i++) {
    CHECK_CUDA(cudaMemcpyAsync(p->dst_addr, p->src_addr, p->cp_bytes,
cudaMemcpyDefault, p->stream1));
  }
  CHECK_CUDA(cudaEventRecord(p->end_event, p->stream1));

  CHECK_CUDA(cudaEventSynchronize(p->end_event));
  float time_in_ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&time_in_ms, p->start_event, p->end_event));

  LOGGER(INFO, "Multi-path[%d] thr GPU:%d BW:%.2f GB/Sec\n", p->path_id,
p->gpu_id, p->cp_bytes * p->iter / time_in_ms / 1e6);

  return nullptr;
}
*/
