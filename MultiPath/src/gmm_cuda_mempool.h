#pragma once

#include <mutex>
#include <unordered_map>

#include "cuda.h"
#include "gmm_cuda_common.h"
#include "gmm_util.h"

typedef int ShareableHandle;

struct dev_attr_check {
  CUdevice_attribute attr;
  int desired_val;
  char desc[64];
};

static dev_attr_check dev_attr_checkList[] = {
    {CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, 1, "Unified Addressing"},
    {CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED, 1, "host register"},
    {CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, 1, "VMM"},
    {CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, 1,
     "VMM IPC "},
    {CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, 1, "VMM+GDR"},
    {CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED, 1, "hostMem as RO"},
    {CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED, 1,
     "timeline semaphore interop"},
    {CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, 1, "GPUDirect"},
    {CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, 1, "mempool"},
    {CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES, 1, "mempool + IPC"},
};

static void preCheck_memPool_support(int dev, dev_attr_check *checkList) {
  int val = 0;

  if (checkList) {
    int check_cnt = sizeof(checkList) / sizeof(checkList[0]);
    for (int i = 0; i < check_cnt; ++i) {
      CHECK_DRV(cuDeviceGetAttribute(&val, checkList[i].attr, dev));
      printf("Check dev.attr:%d val:%d desc:%s %s\n", checkList[i].attr, val,
             checkList[i].desc,
             (val == checkList[i].desired_val) ? "supported" : "NOT supported");
    }
  }
}

struct cuda_lowAlloc_desc {
  size_t orig_size;
  size_t alloc_size;
  bool rw;
  bool export_flag;

 public:
  cuda_lowAlloc_desc(size_t orig_sz, size_t alloc_sz, bool rw_,
                     bool export_flag_)
      : orig_size(orig_sz),
        alloc_size(alloc_sz),
        rw(rw_),
        export_flag(export_flag_) {}
};

// GMM managed mem pool
class gmm_memPool {
 private:
  int cur_dev;
  // stats from explict alloc/free
  size_t tot_bytes;
  size_t free_bytes;
  size_t used_bytes;

  size_t max_bytes;
  size_t min_bytes;  // at least keep those

  int dev_idx;

  int type;          // tiering, backed, ...
  int max_free_pct;  // start trim if free bytes exceed this
  int stats;

  // req list
  std::mutex lock;

  // for each alloc desc: size, RW, export or not
  std::unordered_map<CUdeviceptr, cuda_lowAlloc_desc *> alloc_list;

  // PoC implemented based on CUDA async allocator
  CUmemoryPool pre_pool;
  CUmemPoolProps pool_props;
  ShareableHandle pool_handle;
  CUmemAccessDesc access_desc;
  CUmemAllocationProp alloc_prop;
  size_t alloc_gran;

 public:
  gmm_memPool(int dev, size_t init_sz, size_t max_sz)
      : cur_dev(dev), max_bytes(max_sz) {
    // printf("to create mempool on dev:%d init_sz:%ld max_sz:%ld\n", cur_dev,
    // init_sz, max_sz);

    memset(&access_desc, 0, sizeof(access_desc));
    memset(&pool_props, 0, sizeof(pool_props));
    memset(&alloc_prop, 0, sizeof(alloc_prop));

    {
      alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      alloc_prop.location.id = cur_dev;
      alloc_prop.requestedHandleTypes =
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
      CHECK_DRV(cuMemGetAllocationGranularity(
          &alloc_gran, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    }

    create_pool(cur_dev);
    if (init_sz > 0) {
      CUdeviceptr dptr;
      CUstream str = 0UL;
      if ((0 == alloc(&dptr, init_sz, str)) && (0 == free_dptr(dptr, str))) {
        CHECK_DRV(cuMemPoolSetAttribute(
            pre_pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &init_sz));
        // printf("create mempool on dev:%d init_sz:%ld max_sz:%ld pre-fill
        // done\n", cur_dev, init_sz, max_sz);
      }
    }
  }

  ~gmm_memPool() { destroy_pool(); }

  int create_pool(int dev) {
    int ret = 0;
    pool_props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    pool_props.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    pool_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    pool_props.location.id = dev;

    CHECK_DRV(cuMemPoolCreate(&pre_pool, &pool_props));
    CHECK_DRV((cuMemPoolExportToShareableHandle(
        &pool_handle, pre_pool, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0)));
    return ret;
  }

  void destroy_pool() {
    printf("To destory pre-poool");
    CHECK_DRV(cuMemPoolDestroy(pre_pool));
  }

  // allocator for cur_dev, RW
  int alloc(CUdeviceptr *dptr, size_t bytes, CUstream stream) {
    return alloc(dptr, bytes, stream, cur_dev, true, nullptr, false);
  }

  // allocator with more alloc option
  int alloc(CUdeviceptr *dptr, size_t bytes, CUstream stream, int use_dev,
            bool rw = true, CUmemPoolPtrExportData *export_data = nullptr,
            bool export_flag = false) {
    int ret = 0;
    size_t alloc_size = alignSize(bytes, alloc_gran);

    std::lock_guard<std::mutex> h_lock(lock);
    CHECK_DRV(cuMemAllocFromPoolAsync(dptr, bytes, pre_pool, stream));

    cuda_lowAlloc_desc *alloc_desc =
        new cuda_lowAlloc_desc(rw, bytes, alloc_size, export_flag);

    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = use_dev;
    if (rw) {
      access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    } else {
      access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
    }

    CHECK_DRV(cuMemPoolSetAccess(pre_pool, &access_desc, bytes));

    if (export_flag) {
      memset((void *)export_data, 0, sizeof(*export_data));
      CHECK_DRV(cuMemPoolExportPointer(export_data, *dptr));
    }

    alloc_list[*dptr] = alloc_desc;
    return 0;
  }

  int free_dptr(CUdeviceptr dptr, CUstream stream) {
    CHECK_DRV(cuMemFreeAsync(dptr, stream));

    auto ent = alloc_list.find(dptr);
    if (ent != alloc_list.end()) {
      delete ent->second;
      alloc_list.erase(dptr);
    }
    return 0;
  }

  inline size_t get_bytes() {
    std::lock_guard<std::mutex> h_lock(lock);
    return tot_bytes;
  }

  inline size_t get_freeBytes() {
    std::lock_guard<std::mutex> h_lock(lock);
    return free_bytes;
  }

  inline size_t get_usedBytes() {
    std::lock_guard<std::mutex> h_lock(lock);
    return used_bytes;
  }

  int export_handle();
  int import_handle();
  // trim
  // swap out
};
