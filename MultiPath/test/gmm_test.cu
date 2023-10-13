#include <assert.h>
#include <cuda_runtime.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <iostream>

#include "cuda_check.h"
#include "glake_cache.h"

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
    printf("%s to_verify: %d %d, truth: %d %d\n", __func__, d2[0],
           d2[bytes - 1], d1[0], d1[bytes - 1]);
    free(check_buf);
  } else if (truth_at_cpu && !to_verify_at_gpu) {
    char *d2 = (char *)to_verify;
    char *d1 = (char *)truth;
    printf("%s to_verify: %d %d, truth: %d %d\n", __func__, d2[0],
           d2[bytes - 1], d1[0], d1[bytes - 1]);
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

void cuda_h2d(char *d_ptr, char *h_ptr, size_t bytes, cudaStream_t stream,
              char *truth_ptr, const char *name) {
  printf("[Test %s_%s] start...\n", __func__, name);
  CHECK_CUDA(
      cudaMemcpyAsync(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice, stream));
  cudaStreamSynchronize(stream);
  int ret = verify_data((void *)d_ptr, (void *)truth_ptr, bytes, true, true);
  if (ret == 0) {
    printf("[Test %s_%s] Pass\n", __func__, name);
  } else {
    printf("[Test %s_%s] Fail\n", __func__, name);
  }
  printf("\n");
}

void cuda_d2h(char *h_ptr, char *d_ptr, size_t bytes, cudaStream_t stream,
              char *truth_ptr, const char *name) {
  printf("[Test %s_%s] start...\n", __func__, name);
  CHECK_CUDA(
      cudaMemcpyAsync(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
  int ret = verify_data((void *)h_ptr, (void *)truth_ptr, bytes, true, false);
  if (ret == 0) {
    printf("[Test %s_%s] Pass\n", __func__, name);
  } else {
    printf("[Test %s_%s] Fail\n", __func__, name);
  }
  printf("\n");
}

void multi_path_h2d(char *d_ptr, char *h_ptr, size_t bytes, cudaStream_t stream,
                    char *truth_ptr, const char *name) {
  printf("[Test %s_%s] start...\n", __func__, name);
  // CHECK_CUDA(cudaMemcpyAsync(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice,
  // stream));
  glake::H2DMultiPath(d_ptr, h_ptr, bytes, stream);
  cudaStreamSynchronize(stream);
  int ret = verify_data((void *)d_ptr, (void *)truth_ptr, bytes, true, true);
  if (ret == 0) {
    printf("[Test %s_%s] Pass\n", __func__, name);
  } else {
    printf("[Test %s_%s] Fail\n", __func__, name);
  }
  printf("\n");
}

void multi_path_d2h(char *h_ptr, char *d_ptr, size_t bytes, cudaStream_t stream,
                    char *truth_ptr, const char *name) {
  printf("[Test %s_%s] start...\n", __func__, name);
  // CHECK_CUDA(cudaMemcpyAsync(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost,
  // stream));
  glake::D2HMultiPath(h_ptr, d_ptr, bytes, stream);
  cudaStreamSynchronize(stream);
  int ret = verify_data((void *)h_ptr, (void *)truth_ptr, bytes, true, false);
  if (ret == 0) {
    printf("[Test %s_%s] Pass\n", __func__, name);
  } else {
    printf("[Test %s_%s] Fail\n", __func__, name);
  }
  printf("\n");
}

/*
 * h_mem_type: 0=cudaMallocHost, 1=malloc+cudaHostRegister, 2=malloc.
 */
static void test_multi_path(size_t bytes, int nIter, int h_mem_type) {
  printf("\n==============================================\n");
  // Start of Init.
  // h_A is truth value, unmodifiable.
  char *h_A = nullptr;
  char *h_B = nullptr;
  char *d_A = nullptr;
  char *d_B = nullptr;
  if (h_mem_type == 0) {
    CHECK_CUDA(cudaMallocHost(&h_A, bytes));
    CHECK_CUDA(cudaMallocHost(&h_B, bytes));
  } else if (h_mem_type == 1) {
    h_A = (char *)malloc(bytes);
    h_B = (char *)malloc(bytes);
    // CHECK_CUDA(cudaHostRegister(h_A, bytes, cudaHostRegisterDefault));
    // CHECK_CUDA(cudaHostRegister(h_B, bytes, cudaHostRegisterDefault));
    CHECK_DRV(cuMemHostRegister(
        h_A, bytes,
        CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP));
    CHECK_DRV(cuMemHostRegister(
        h_B, bytes,
        CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP));
  } else if (h_mem_type == 2) {
    h_A = (char *)malloc(bytes);
    h_B = (char *)malloc(bytes);
  } else {
    printf("Error h_mem_type:%d\n", h_mem_type);
    return;
  }

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

  // printf("Start SG test size:%ld iter:%d host:%p dev:%p dev2:%p\n", bytes,
  // nIter, h_A, d_A, d_B);
  CHECK_CUDA(cudaMemcpyAsync((char *)d_A, (char *)h_A, bytes,
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  // End of Init.

  cuda_d2h(h_B, d_A, bytes, stream, h_A, "base");
  // multi_path_d2h(h_B, d_A, bytes, stream, h_A, "base");

  cudaMemset(d_A, 0, bytes);
  cudaMemset(d_B, 0, bytes);
  memset(h_B, 0, bytes);
  cuda_h2d(d_B, h_A, bytes, stream, h_A, "base");
  // multi_path_h2d(d_B, h_A, bytes, stream, h_A, "base");

  const size_t half_bytes = bytes / 2;
  char *h_A2 = h_A + half_bytes;
  char *d_A2 = d_A + half_bytes;
  char *d_B2 = d_B + half_bytes;
  char *h_B2 = h_B + half_bytes;
  cudaMemset(d_A, 0, bytes);
  cudaMemset(d_B, 0, bytes);
  memset(h_B, 0, bytes);

  cudaMemcpy(d_A2, h_A2, half_bytes, cudaMemcpyHostToDevice);
  cuda_d2h(h_B2, d_A2, half_bytes, stream, h_A2, "half");
  // multi_path_d2h(h_B2, d_A2, half_bytes, stream, h_A2, "half");

  cudaMemset(d_A, 0, bytes);
  cudaMemset(d_B, 0, bytes);
  memset(h_B, 0, bytes);
  cuda_h2d(d_B2, h_A2, half_bytes, stream, h_A2, "half");
  // multi_path_h2d(d_B2, h_A2, half_bytes, stream, h_A2, "half");

  CHECK_CUDA(cudaFree((void *)d_A));
  CHECK_CUDA(cudaFree((void *)d_B));
  CHECK_CUDA(cudaStreamDestroy(stream));
  if (h_mem_type == 0) {
    CHECK_CUDA(cudaFreeHost(h_A));
    CHECK_CUDA(cudaFreeHost(h_B));
  } else if (h_mem_type == 1) {
    CHECK_CUDA(cudaHostUnregister(h_A));
    CHECK_CUDA(cudaHostUnregister(h_B));
    free(h_A);
    free(h_B);
  } else if (h_mem_type == 2) {
    free(h_A);
    free(h_B);
  } else {
    printf("Error h_mem_type:%d\n", h_mem_type);
  }
  return;
}

int main(int argc, char *argv[]) {
  size_t N = getenv("N") ? atol(getenv("N")) : 2048UL;  // test size in MB
  int nIter = getenv("I") ? atoi(getenv("I")) : 100;    // test iterations

  size_t bytes = (N << 20);

  enable_p2p_full();

  test_multi_path(bytes, nIter, 0);
  // test_multi_path(bytes, nIter, 1);
  // test_multi_path(bytes, nIter, 2);

  printf("Test complete\n");
  return 0;
}
