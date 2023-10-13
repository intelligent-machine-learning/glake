#include <assert.h>
#include <cuda_runtime.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <iostream>

#include "cuda_check.h"

static int verify_data(void *const &to_verify, void *const &truth, size_t bytes,
                       bool truth_at_cpu = true, bool to_verify_at_gpu = true) {
  int ret = -1;

  if (truth_at_cpu && to_verify_at_gpu) {
    char *check_buf = (char *)calloc(bytes, sizeof(char));
    CHECK_CUDA(cudaMemcpy(check_buf, to_verify, bytes, cudaMemcpyDeviceToHost));
    if (((char *)check_buf)[0] == 0) {
      printf("Value should not be 0\n");
      return -1;
    }
    ret = memcmp(truth, check_buf, bytes);
    char *d2 = (char *)check_buf;
    char *d1 = (char *)truth;
    //printf("%s to_verify: %d %d, truth: %d %d\n", __func__, d2[0],
    //       d2[bytes - 1], d1[0], d1[bytes - 1]);
    free(check_buf);
  } else if (truth_at_cpu && !to_verify_at_gpu) {
    char *d2 = (char *)to_verify;
    char *d1 = (char *)truth;
    //printf("%s to_verify: %d %d, truth: %d %d\n", __func__, d2[0],
    //       d2[bytes - 1], d1[0], d1[bytes - 1]);
    if (((char *)to_verify)[0] == 0) {
      printf("%s Value should not be 0", __func__);
      return -1;
    }
    ret = memcmp(to_verify, truth, bytes);
  } else {
    printf("%s Error truth_at_cpu:%d to_verify_at_gpu:%d", __func__,
           truth_at_cpu, to_verify_at_gpu);
  }

  // printf("Verify result: %s\n", ret? "failed":"ok");
  return ret;
}

static void enable_p2p_full() {
  int devCnt;
  CHECK_CUDA(cudaGetDeviceCount(&devCnt));

  for (int i = 0; i < devCnt; i++) {
    CHECK_CUDA(cudaSetDevice(i));

    for (int j = i; j < devCnt; j++) {
      if (i == j) continue;
      int accessSupported = 0;
      CHECK_CUDA(cudaDeviceGetP2PAttribute(
          &accessSupported, cudaDevP2PAttrAccessSupported, i, j));
      if (accessSupported) {
        (cudaDeviceEnablePeerAccess(j, 0));
        CHECK_CUDA(cudaSetDevice(j));
        (cudaDeviceEnablePeerAccess(i, 0));
      }
    }
  }
}

// Got bandwidth 83.1 GB/s on a platform: 8 * A100(80GB) with NvLink, 4 PCIe
// paths between CPU-GPU.
void cuda_h2d(char *d_ptr, char *h_ptr, size_t bytes, int iter,
              cudaStream_t stream, char *truth_ptr, const char *name) {
  printf("[Test %s_%s] start...\n", __func__, name);

  // Warm up
  CHECK_CUDA(
      cudaMemcpyAsync(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice, stream));
  cudaStreamSynchronize(stream);

  auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < iter; ++i) {
    CHECK_CUDA(
        cudaMemcpyAsync(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice, stream));
  }
  cudaStreamSynchronize(stream);
  auto t2 = std::chrono::steady_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << "H2D: " << dur.count() << " us, Bytes:" << (bytes >> 20)
            << " MB, Bandwidth:" << (float)(bytes >> 10) * iter / dur.count()
            << " GB/s\n";

  int ret = verify_data((void *)d_ptr, (void *)truth_ptr, bytes, true, true);
  if (ret == 0) {
    printf("[Test %s_%s] Pass\n", __func__, name);
  } else {
    printf("[Test %s_%s] Fail\n", __func__, name);
  }
  printf("\n");
}

// Got bandwidth 71.2 GB/s on a platform: 8 * A100(80GB) with NvLink, 4 PCIe
// paths between CPU-GPU.
void cuda_d2h(char *h_ptr, char *d_ptr, size_t bytes, int iter,
              cudaStream_t stream, char *truth_ptr, const char *name) {
  printf("[Test %s_%s] start...\n", __func__, name);

  // Warm up
  CHECK_CUDA(
      cudaMemcpyAsync(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < iter; ++i) {
    CHECK_CUDA(
        cudaMemcpyAsync(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost, stream));
  }
  cudaStreamSynchronize(stream);
  auto t2 = std::chrono::steady_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << "D2H: " << dur.count() << " us, Bytes:" << (bytes >> 20)
            << " MB, Bandwidth:" << (float)(bytes >> 10) * iter / dur.count()
            << " GB/s\n";

  int ret = verify_data((void *)h_ptr, (void *)truth_ptr, bytes, true, false);
  if (ret == 0) {
    printf("[Test %s_%s] Pass\n", __func__, name);
  } else {
    printf("[Test %s_%s] Fail\n", __func__, name);
  }
  printf("\n");
}

static void bench_multi_path(size_t bytes, int nIter) {
  printf("\n==============================================\n");
  // Start of Init.
  // h_A is truth value, unmodifiable.
  char *h_A = nullptr;
  char *h_B = nullptr;
  char *d_A = nullptr;
  char *d_B = nullptr;
  CHECK_CUDA(cudaMallocHost(&h_A, bytes));
  CHECK_CUDA(cudaMallocHost(&h_B, bytes));
  for (size_t i = 0; i < bytes; i++) {
    h_A[i] = rand() / static_cast<char>(RAND_MAX);
    h_B[i] = 0;
  }

  int devCnt = 0, cur_dev = 0;
  CHECK_CUDA(cudaGetDeviceCount(&devCnt));
  size_t free_sz = 0, tot_sz = 0;
  CHECK_CUDA(cudaMemGetInfo(&free_sz, &tot_sz));

  CHECK_CUDA(cudaSetDevice(cur_dev));
  CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
  CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  CHECK_CUDA(cudaMemcpyAsync((char *)d_A, (char *)h_A, bytes,
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  // End of Init.

  // Benchmark D2H copy.
  cuda_d2h(h_B, d_A, bytes, nIter, stream, h_A, "base");

  cudaMemset(d_A, 0, bytes);
  cudaMemset(d_B, 0, bytes);
  memset(h_B, 0, bytes);
  // Benchmark H2D copy.
  cuda_h2d(d_B, h_A, bytes, nIter, stream, h_A, "base");

  CHECK_CUDA(cudaFree((void *)d_A));
  CHECK_CUDA(cudaFree((void *)d_B));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFreeHost(h_A));
  CHECK_CUDA(cudaFreeHost(h_B));
  return;
}

int main(int argc, char *argv[]) {
  size_t N = getenv("N") ? atol(getenv("N")) : 2048UL;  // test size in MB
  int nIter = getenv("I") ? atoi(getenv("I")) : 100;    // test iterations

  size_t bytes = (N << 20);

  enable_p2p_full();

  bench_multi_path(bytes, nIter);

  printf("Test complete\n");
  return 0;
}
