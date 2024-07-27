#include "gmm_mempool.h"
#include "gmm_util.h"

gmm_memPool::gmm_memPool(size_t init_MB, size_t max_MB, int dev = 0,
                         size_t min_MB = 128, int free_pct = 50) {
  cudaMemPoolProps props;
  props.allocType = cudaMemAllocationTypePinned;
  props.handleTypes = cudaMemHandleTypePosixFileDescriptor;
  props.location.type = cudaMemLocationTypeDevice;
  props.location.id = dev;

  CHECK_CUDA(cudaMemPoolCreate(&memPool, props));
  LOGGER(INFO, "create memPool done");

  size_t bytes = init_sz >> 20;
  int *ptr;

  CHECK_CUDA(cudaMallocFromPoolAsync((void **)&ptr, bytes, memPool, 0));
  if (ptr) {
    // keep at least min_MB
    CHECK_CUDA(cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, min_MB >> 20);

    LOGGER(INFO, "alloc %ld from memPool", bytes);
    CHECK_CUDA(cudaMemSet(ptr, bytes, memPool, 0));
    LOGGER(INFO, "set mem done");
    CHECK_CUDA(cudaFreeAsync(ptr, 0));

    size_t curr_resv, curr_used;
    CHECK_CUDA(cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemCurrent, &curr_resv));
    CHECK_CUDA(cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemCurrent, &curr_used));
    LOGGER(INFO, "mempool resv:%ld used:%ld", curr_resv, curr_used);

    //TODO: double check input	
    tot_bytes  = bytes;
    free_bytes = bytes;
    used_bytes = 0;

    max_bytes = max_sz  >> 20;
    min_bytes = min_MB  >> 20;
    max_free_pct = free_pct;
    dev_idx = dev;
    // uint64_t threshold = UINT64_MAX;
  } else {
    tot_bytes = 0;
    free_bytes = 0;
    used_bytes = 0;
    max_bytes = 0;
    min_bytes = 0;
    max_free_pct = 0;
    dev_idx = -1;
  }
}

gmm_memPool::~gmm_memPool() {
  // TODO: drain req
  CHECK_CUDA(cudaMemPoolDestroy(memPool));
  LOGGER(INFO, "destroy mempool done");
}

// alloc from memPool
int alloc(void **ptr, size_t bytes, cudaStream_t stream = 0) {
  // memory pool may be from a device different than that of the specified
  // hStream.
  CHECK_CUDA(cudaMallocFromPoolAsync(ptr, bytes, memPool, stream));
  LOGGER(INFO, "alloc %ld from pool");

  alloc_list.insert(std::make_pair(*ptr, bytes));

  {
    std::lock_guard<std::mutex> h_lock(lock);
    used_bytes += bytes;

    // means trigger extra alloc
    if (used_bytes > tot_bytes) size_t curr_resv;
    CHECK_CUDA(cudaMemPoolGetAttribute(
        memPool, cudaMemPoolAttrReservedMemCurrent, &curr_resv));
    tot_bytes = curr_resv;

    free_bytes = 0;  // ? all alloc out
  }
}

if (max_bytes > 0) {
  size_t curr_resv;
  CHECK_CUDA(cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemCurrent,
                                     &curr_resv));
  if (curr_resv > max_bytes)
    LOGGER(WARNING, "mempool tot resv bytes:%ld already exceed max set:%ld",
           curr_resv, max_bytes);
}

return (ptr) ? 0 : 1;
}

// free ptr
int free(void *ptr, cudaStream_t stream = 0) {
  CHECK_CUDA(cudaFreeAsync(ptr, stream));
  LOGGER(INFO, "try to free %p to pool", ptr);

  auto it = alloc_list.find(ptr);
  if (it != alloc_list.end()) {
    size_t bytes = alloc_list[ptr];
    LOGGER(INFO, "get size:%ld for %p", bytes, ptr);
    alloc_list.erase(ptr);

    {
      std::lock_guard<std::mutex> h_lock(lock);
      free_bytes += bytes;
      used_bytes -= bytes;
    }

    if (free_bytes > tot_bytes * max_free_pct / 100) {
      LOGGER(WARNING, "mempool tot:%ld free:%ld takes:%.2f exceed:%d%%",
             tot_bytes, free_bytes, free_bytes * 1.0 / tot_bytes, max_free_pct);
    }
  }
}

// cudaMemPoolGetAttribute (cudaMemPool_t memPool, cudaMemPoolAttr attr, void*
// value) cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold,
// &threshold);

/*
cudaMallocAsync (void** devPtr, size_t size, cudaStream_t hStream)
cudaFreeAsync(*dptr, hStream)

cudaMemPoolExportPointer (cudaMemPoolPtrExportData* exportData, void* ptr)
cudaMemPoolExportToShareableHandle (void* shareableHandle, cudaMemPool_t
memPool, cudaMemAllocationHandleType handleType, unsigned int flags)

cudaMemPoolGetAccess (cudaMemAccessFlags ** flags, cudaMemPool_t memPool,
cudaMemLocation* location)

cudaMemPoolImportFromShareableHandle (cudaMemPool_t* memPool, void*
shareableHandle, cudaMemAllocationHandleType handleType, unsigned int flags)
cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep)
*/
