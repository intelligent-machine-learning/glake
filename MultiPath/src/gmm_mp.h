#pragma once
// Unroll depth in SM copy kernel
const int NUM_LOOP_UNROLL = 2;

// Thread block size
const int NUM_THREADS_IN_BLOCK = 128;

__global__ void SMCopyKernel(ulong2 *tgt, const ulong2 *src);
int setGpu(int gpu_id);

// void * htod_direct(void *args);
// void * dtoh_direct(void *args);

int gmm_launch_cp_kernel(char *tgt_addr, char *src_addr, size_t bytes,
                         cudaStream_t stream, CUevent &pre_evt,
                         CUevent &post_evt, bool sync = false);

int gmm_launch_cp_DMA(char *tgt_addr, char *src_addr, char *tmp_buf,
                      size_t bytes, size_t buf_bytes, cudaStream_t stream,
                      CUevent &pre_evt, CUevent &post_evt, bool sync);

int gmm_DMA_direct(char *tgt_addr, char *src_addr, size_t bytes,
                   cudaStream_t stream, CUevent &pre_evt, CUevent &post_evt,
                   bool sync);

int gmm_pipeline_DMA(char *tgt_addr, char *src_addr, size_t bytes,
                     char *tmp_buf1, char *tmp_buf2, size_t buf_bytes,
                     cudaStream_t pipe_stream1, cudaStream_t pipe_stream2,
                     cudaEvent_t &pipe_evt1, cudaEvent_t &pipe_evt2,
                     CUevent &pre_evt, CUevent &post_evt, bool sync);

int gmm_verify_data(void *const &to_verify, void *const &truth, size_t bytes,
                    bool baseline_at_cpu, bool to_verify_at_gpu);
