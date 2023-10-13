#pragma once

#include <cstdint>
#include "gmm_common.h"
#include "gmm_api_stats.h"

// global config and api stats per client process
// could be used indepdently with GMM other perf optimizaitons, such as purely
// trace
enum gmm_mode_ {
  GMM_MEM_MODE_DEFAULT = 0,
  GMM_MEM_MODE_ORIG = 1,
  GMM_MEM_MODE_UM = 2,
  GMM_MEM_MODE_GLOBAL = 3,
  GMM_MEM_MODE_ASYNC = 4,
  GMM_MEM_MODE_VMM = 5
};

struct gmm_client_cfg {
  int pid;
  int dev_cnt;
  int GMM_MEM_MODE;
  int GMM_MP;
  int UM_GB;
  int OOM_HOSTALLOC;
  uint32_t GMM_GDR_MAX_SZ;
  int cuda_back;

  volatile int xgpu_refCnt;
  size_t mem_alloc_size[MAX_DEV_NR];

  api_stats api_stat;

 public:
  gmm_client_cfg() {
    pid = getpid();
    xgpu_refCnt = 0;
    dev_cnt = 0;
    GMM_MEM_MODE = getenv("GMM_MEM_MODE") ? atoi(getenv("GMM_MEM_MODE"))
                                          : GMM_MEM_MODE_DEFAULT;
    GMM_MP = getenv("GMM_MP") ? atoi(getenv("GMM_MP")) : 1;
    UM_GB = getenv("UM_GB") ? atoi(getenv("UM_GB")) : 32;
    cuda_back = getenv("CUDA_BACK") ? atoi(getenv("CUDA_BACK")) : 0;
    OOM_HOSTALLOC = getenv("OOM_HOSTALLOC") ? atoi(getenv("OOM_HOSTALLOC")) : 1;
    GMM_GDR_MAX_SZ = getenv("GMM_GDR_MAX_SZ") ? atoi(getenv("GMM_GDR_MAX_SZ"))
                                              : GMM_GDR_MAX_SZ_DEFAULT;
  }

  ~gmm_client_cfg() {}

 public:
  int get_memMode() { return GMM_MEM_MODE; }
  int get_MP() { return GMM_MP; }
  int get_UM_GB() { return UM_GB; }
  int get_cuda_back() { return cuda_back; }
  int get_OOM_HOSTALLOC() { return OOM_HOSTALLOC; }
  uint32_t get_GDR_max_sz() { return GMM_GDR_MAX_SZ; }

  size_t get_alloc_size(int dev) { return mem_alloc_size[dev]; }
  void dec_alloc_size(int dev, size_t delta) { mem_alloc_size[dev] -= delta; }
  void inc_alloc_size(int dev, size_t delta) { mem_alloc_size[dev] += delta; }
};

// hook orig libcuda, init cfg and stats when loading lib
void gmm_client_cfg_init(void *&libP, gmm_client_cfg *&cfg);
void gmm_client_cfg_destroy(void *libP);
