#include <cuda.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <unordered_map>

#include "glake_cache.h"
#include "gmm_client.h"
#include "gmm_client_cfg.h"
#include "gmm_common.h"
#include "gmm_gdr_plugin.h"

///////////////////////////////////////////////////////////
#define LOGAPI() LOGD("%s dev:%d", __FUNCTION__, __GETDEV())
#ifdef __cplusplus
extern "C" {
#endif

void *libP = NULL;

static gmm_client_cfg *g_cfg;
static gmm_client_ctx *gmm_ctx_p;

static std::mutex gmm_lock;
static std::atomic<bool> init_topo(false);

void InitClientCtx() {
  if (!init_topo) {
    init_topo = true;
    std::lock_guard<std::mutex> lock_(gmm_lock);
    if (g_cfg->get_MP() && !gmm_ctx_p) {  // defer init until first mem alloc
      gmm_ctx_p = new gmm_client_ctx(g_cfg);
    }
  }
}

static void __attribute__((constructor)) x_init(void) {
  gmm_client_cfg_init(libP, g_cfg);
  // LOGI("after cfg_init, libP:%p", libP);
}

static void __attribute__((destructor)) x_fini(void) {
  gmm_client_cfg_destroy(libP);
}

//////////////////////////////////////////////
#if defined(MODULE_STATUS)
#undef MODULE_STATUS
#define MODULE_STATUS CUresult
#else
#define MODULE_STATUS CUresult
#endif

// By default, set false.
static bool IsMultiPath() {
  const char *env = std::getenv("GLAKE_MULTI_PATH");
  if (!env) {
    return false;
  } else {
    int env2 = std::stoi(env);
    if (env2 == 1) {
      // printf("[Debug] Env GLAKE_MULTI_PATH=%d. Set true!\n", env2);
      return true;
    } else if (env2 == 0) {
      // printf("[Debug] Env GLAKE_MULTI_PATH=%d. Set false!\n", env2);
      return false;
    } else {
      printf("[Warn] Env GLAKE_MULTI_PATH=%d unknown. Set false!\n", env2);
      return false;
    }
  }
}

CUresult CUDAAPI cuGetErrorString(CUresult error, const char **pStr) {
  __C()(error, pStr);
}

CUresult CUDAAPI cuGetErrorName(CUresult error, const char **pStr) {
  __C()(error, pStr);
}

CUresult CUDAAPI cuInit(unsigned int Flags) {
  CUresult err_ = __CF("cuInit")(Flags);
  if (err_ != CUDA_SUCCESS) {
    LOGW("%d %s return err:%d\n", getpid(), __FUNCTION__, err_);
  }

  return err_;
}

CUresult CUDAAPI cuDriverGetVersion(int *driverVersion) {
  __C()(driverVersion);
}

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
  __C()(device, ordinal);
}

CUresult CUDAAPI cuDeviceGetCount(int *count) { __C()(count); }

CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev) {
  __C()(name, len, dev);
}

CUresult CUDAAPI cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
  __C()(uuid, dev);
}

CUresult CUDAAPI cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) {
  __C()(uuid, dev);
}

CUresult CUDAAPI cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask,
                                 CUdevice dev) {
  __C()(luid, deviceNodeMask, dev);
}

CUresult CUDAAPI cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
  if (0 || g_cfg->get_memMode() == g_cfg->get_memMode()) {
    *bytes = (g_cfg->get_UM_GB() << 30);
    return CUDA_SUCCESS;
  }
  __C()(bytes, dev);
}

CUresult CUDAAPI cuDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements,
                                                    CUarray_format format,
                                                    unsigned numChannels,
                                                    CUdevice dev) {
  __C()(maxWidthInElements, format, numChannels, dev);
}

CUresult CUDAAPI cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib,
                                      CUdevice dev) {
  __C()(pi, attrib, dev);
}

CUresult CUDAAPI cuDeviceGetNvSciSyncAttributes(void *nvSciSyncAttrList,
                                                CUdevice dev, int flags) {
  __C()(nvSciSyncAttrList, dev, flags);
}

CUresult CUDAAPI cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) {
  __C()(dev, pool);
}

CUresult CUDAAPI cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev) {
  __C()(pool, dev);
}

CUresult CUDAAPI cuDeviceGetDefaultMemPool(CUmemoryPool *pool_out,
                                           CUdevice dev) {
  __C()(pool_out, dev);
}

CUresult CUDAAPI
cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target,
                           CUflushGPUDirectRDMAWritesScope scope) {
  __C()(target, scope);
}

CUresult CUDAAPI cuDeviceGetProperties(CUdevprop *prop, CUdevice dev) {
  __C()(prop, dev);
}

CUresult CUDAAPI cuDeviceComputeCapability(int *major, int *minor,
                                           CUdevice dev) {
  __C()(major, minor, dev);
}

CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
  CUresult rst = __CF("cuDevicePrimaryCtxRetain")(pctx, dev);
  // LOGI("pid:%d tid:%d retain ctx on dev:%d ctx:%p", getpid(), gettid(), dev,
  // pctx);
  return rst;
}

CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev) { __C()(dev); }

CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
  __C()(dev, flags);
}

CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags,
                                            int *active) {
  __C()(dev, flags, active);
}

CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice dev) { __C()(dev); }

CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
  __C()(pctx, flags, dev);
}

CUresult cuCtxCreate_v3(CUcontext *pctx, CUexecAffinityParam *paramsArray,
                        int numParams, unsigned int flags, CUdevice dev) {
  __C()(pctx, paramsArray, numParams, flags, dev);
}

CUresult cuCtxDestroy(CUcontext ctx) { __C()(ctx); }

CUresult cuCtxPushCurrent(CUcontext ctx) {
  LOGI("push new ctx");
  __I("cuCtxPushCurrent_v2")(ctx);
}

/*
CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
  __C()(pctx, flags, dev);
}
CUresult cuCtxDestroy_v2(CUcontext ctx)
{
  __I("cuCtxDestroy_v2")(ctx);
}
CUresult cuCtxPushCurrent_v2(CUcontext ctx)
{
  __I("cuCtxPushCurrent_v2")(ctx);
}
CUresult cuCtxPopCurrent_v2(CUcontext *pctx)
{
  __I("cuCtxPopCurrent_v2")(pctx);
}
*/
CUresult cuCtxPopCurrent(CUcontext *pctx) { __C()(pctx); }

CUresult cuCtxSetCurrent(CUcontext ctx) { __C()(ctx); }

CUresult cuCtxGetCurrent(CUcontext *pctx) { __C()(pctx); }

CUresult cuCtxGetDevice(CUdevice *device) { __C()(device); }

CUresult cuCtxGetFlags(unsigned int *flags) { __C()(flags); }

CUresult CUDAAPI cuCtxSynchronize(void) { __C()(); }

CUresult CUDAAPI cuCtxSetLimit(CUlimit limit, size_t value) {
  __C()(limit, value);
}

CUresult CUDAAPI cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
  __C()(pvalue, limit);
}

CUresult CUDAAPI cuCtxGetCacheConfig(CUfunc_cache *pconfig) { __C()(pconfig); }

CUresult CUDAAPI cuCtxSetCacheConfig(CUfunc_cache config) { __C()(config); }

CUresult CUDAAPI cuCtxGetSharedMemConfig(CUsharedconfig *pConfig) {
  __C()(pConfig);
}

CUresult CUDAAPI cuCtxSetSharedMemConfig(CUsharedconfig config) {
  __C()(config);
}

CUresult CUDAAPI cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
  __C()(ctx, version);
}

CUresult CUDAAPI cuCtxGetStreamPriorityRange(int *leastPriority,
                                             int *greatestPriority) {
  __C()(leastPriority, greatestPriority);
}

CUresult CUDAAPI cuCtxResetPersistingL2Cache(void) { __C()(); }

CUresult CUDAAPI cuCtxGetExecAffinity(CUexecAffinityParam *pExecAffinity,
                                      CUexecAffinityType type) {
  __C()(pExecAffinity, type);
}

CUresult CUDAAPI cuCtxAttach(CUcontext *pctx, unsigned int flags) {
  __C()(pctx, flags);
}

CUresult CUDAAPI cuCtxDetach(CUcontext ctx) { __C()(ctx); }

CUresult CUDAAPI cuModuleLoad(CUmodule *module, const char *fname) {
  __C()(module, fname);
}

CUresult CUDAAPI cuModuleLoadData(CUmodule *module, const void *image) {
  __C()(module, image);
}

CUresult CUDAAPI cuModuleLoadDataEx(CUmodule *module, const void *image,
                                    unsigned int numOptions,
                                    CUjit_option *options,
                                    void **optionValues) {
  __C()(module, image, numOptions, options, optionValues);
}

CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
  __C()(module, fatCubin);
}

CUresult CUDAAPI cuModuleUnload(CUmodule hmod) { __C()(hmod); }

CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                                     const char *name) {
  return __CF("cuModuleGetFunction")(hfunc, hmod, name);
}

CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes,
                                   CUmodule hmod, const char *name) {
  return __CF("cuModuleGetGlobal_v2")(dptr, bytes, hmod, name);
}

CUresult CUDAAPI cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod,
                                   const char *name) {
  __C()(pTexRef, hmod, name);
}

CUresult CUDAAPI cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod,
                                    const char *name) {
  __C()(pSurfRef, hmod, name);
}

// TODO
CUresult CUDAAPI cuMemGetInfo(size_t *free, size_t *total) {
  if (g_cfg->get_memMode() == GMM_MEM_MODE_UM) {
    *total = g_cfg->get_UM_GB() << 30;

    CUdevice dev = 0;
    __CF("cuCtxGetDevice")(&dev);

    *free = *total - g_cfg->get_alloc_size(dev);
  }
  __C()(free, total);
}

CUresult CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
  InitClientCtx();

  CUdevice cur_dev = 0;
  __CF("cuCtxGetDevice")(&cur_dev);
  if (gmm_ctx_p && (gmm_ctx_p->devMem_alloc(cur_dev, dptr, bytesize) == 0)) {
    LOGD("pid:%d GPU dev:%d mem alloc bytes:%ld done", getpid(), cur_dev,
         bytesize);
    return CUDA_SUCCESS;
  }

  LOGD("pid:%d GPU dev:%d mem alloc bytes:%ld done", getpid(), cur_dev,
       bytesize);
  return __CF("cuMemAlloc_v2")(dptr, bytesize);
}

CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch,
                                 size_t WidthInBytes, size_t Height,
                                 unsigned int ElementSizeBytes) {
  __C()(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
}

CUresult CUDAAPI cuMemFree(CUdeviceptr dptr) {
  if (dptr == 0) return CUDA_SUCCESS;

  CUdevice cur_dev = 0;
  __CF("cuCtxGetDevice")(&cur_dev);
  if (gmm_ctx_p && (0 == gmm_ctx_p->devMem_free(cur_dev, dptr))) {
    return CUDA_SUCCESS;
  }

  int ret = __CF("cuMemFree_v2")(dptr);
  return (CUresult)ret;
}

CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize,
                                      CUdeviceptr dptr) {
  __C()(pbase, psize, dptr);
}

CUresult CUDAAPI cuMemAllocHost(void **pp, size_t bytesize) {
  return cuMemHostAlloc(pp, bytesize,
                        CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP);
}

CUresult CUDAAPI cuMemFreeHost(void *p) {
  int ret = 0;
  if (gmm_ctx_p && ((ret = gmm_ctx_p->hostMem_free(p)) == 0)) {
    return CUDA_SUCCESS;
  }

  __C()(p);
}

// TODO: same for cuMemAllocHost
// free by cuMemFreeHost()
CUresult CUDAAPI cuMemHostAlloc(void **pp, size_t bytesize,
                                unsigned int Flags) {
  InitClientCtx();

  CUdevice cur_dev = 0;
  __CF("cuCtxGetDevice")(&cur_dev);
  if (gmm_ctx_p &&
      (gmm_ctx_p->hostMem_alloc(cur_dev, pp, bytesize, Flags) == 0)) {
    return CUDA_SUCCESS;
  }

  CUresult rst = __CF("cuMemHostAlloc")(
      pp, bytesize, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP);
  // LOGD("HostAlloc:%p bytes:%ld flag:%d client:%p", *pp, bytesize, Flags,
  // gmm_ctx_p);
  return rst;
}

CUresult CUDAAPI cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p,
                                           unsigned int Flags) {
  __C()(pdptr, p, Flags);
}

CUresult CUDAAPI cuMemHostGetFlags(unsigned int *pFlags, void *p) {
  __C()(pFlags, p);
}

CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize,
                                   unsigned int flags) {
  __C()(dptr, bytesize, flags);
}

CUresult CUDAAPI cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
  __C()(dev, pciBusId);
}

CUresult CUDAAPI cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
  __C()(pciBusId, len, dev);
}

CUresult CUDAAPI cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event) {
  __C()(pHandle, event);
}

CUresult CUDAAPI cuIpcOpenEventHandle(CUevent *phEvent,
                                      CUipcEventHandle handle) {
  __C()(phEvent, handle);
}

CUresult CUDAAPI cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr) {
  __C()(pHandle, dptr);
}

CUresult CUDAAPI cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle,
                                    unsigned int Flags) {
  __C()(pdptr, handle, Flags);
}

CUresult CUDAAPI cuIpcCloseMemHandle(CUdeviceptr dptr) { __C()(dptr); }

CUresult CUDAAPI cuMemHostRegister(void *p, size_t bytesize,
                                   unsigned int Flags) {
  InitClientCtx();

  host_mem *ent =
      new host_mem(CPU_DEV, p, bytesize, bytesize, HOST_MEM_TYPE_PINNED);
  if (ent) {
    gmm_ctx_p->add_hostMemEntry(ent);
  } else {
    printf("[libcuda] %s new host_mem fail\n", __func__);
  }

  __C()(p, bytesize, Flags);
}

CUresult CUDAAPI cuMemHostUnregister(void *p) { __C()(p); }

CUresult CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
  int src_dev, dst_dev;
  cuPointerGetAttribute(&src_dev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, src);
  cuPointerGetAttribute(&dst_dev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, dst);

  if (src_dev == -1 && dst_dev > 0) {
    return cuMemcpyHtoD_v2(dst, (void *)src, ByteCount);
  } else if (dst_dev == -1 && src_dev > 0) {
    return cuMemcpyDtoH_v2((void *)dst, src, ByteCount);
  } else if (src_dev > 0 && dst_dev > 0) {
    return cuMemcpyDtoD_v2(dst, src, ByteCount);
  } else {
    __C()(dst, src, ByteCount);
  }
}

CUresult CUDAAPI cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                              CUdeviceptr srcDevice, CUcontext srcContext,
                              size_t ByteCount) {
  __C()(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
}

/*
CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t
ByteCount)
{
  __C()(dstDevice, srcHost, ByteCount);
}

CUresult CUDAAPI cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t
ByteCount)
{
  __C()(dstHost, srcDevice, ByteCount);
}

CUresult CUDAAPI cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
size_t ByteCount)
{
  __C()(dstDevice, srcDevice, ByteCount);
}

CUresult CUDAAPI cuMemcpyDtoA(CUarray dstArray, size_t dstOffset, CUdeviceptr
srcDevice, size_t ByteCount)
{
  __C()(dstArray, dstOffset, srcDevice, ByteCount);
}

CUresult CUDAAPI cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray srcArray, size_t
srcOffset, size_t ByteCount)
{
  __C()(dstDevice, srcArray, srcOffset, ByteCount);
}

CUresult CUDAAPI cuMemcpyHtoA(CUarray dstArray, size_t dstOffset, const void
*srcHost, size_t ByteCount)
{
  __C()(dstArray, dstOffset, srcHost, ByteCount);
}

CUresult CUDAAPI cuMemcpyAtoH(void *dstHost, CUarray srcArray, size_t srcOffset,
size_t ByteCount)
{
  __C()(dstHost, srcArray, srcOffset, ByteCount);
}

CUresult CUDAAPI cuMemcpyAtoA(CUarray dstArray, size_t dstOffset, CUarray
srcArray, size_t srcOffset, size_t ByteCount)
{
  __C()(dstArray, dstOffset, srcArray, srcOffset, ByteCount);
}

*/

CUresult CUDAAPI cuMemcpy3D(const CUDA_MEMCPY3D *pCopy) { __C()(pCopy); }

CUresult CUDAAPI cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy) {
  __C()(pCopy);
}

CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                               size_t ByteCount, CUstream hStream) {
  int src_dev, dst_dev;
  cuPointerGetAttribute(&src_dev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, src);
  cuPointerGetAttribute(&dst_dev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, dst);

  if (src_dev == -1 && dst_dev > 0) {
    return __CF("cuMemcpyHtoDAsync_v2")(dst, src, ByteCount, hStream);
  } else if (dst_dev == -1 && src_dev > 0) {
    return __CF("cuMemcpyDtoHAsync_v2")(dst, src, ByteCount, hStream);
  } else if (src_dev > 0 && dst_dev > 0) {
    return __CF("cuMemcpyDtoDAsync_v2")(dst, src, ByteCount, hStream);
  } else {
    LOGI("Async srcdev:%d dst:%d HtoH?", src_dev, dst_dev);
    __C()(dst, src, ByteCount, hStream);
  }
}

CUresult CUDAAPI cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                                   CUdeviceptr srcDevice, CUcontext srcContext,
                                   size_t ByteCount, CUstream hStream) {
  int src_dev, dst_dev;
  cuPointerGetAttribute(&src_dev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                        srcDevice);
  cuPointerGetAttribute(&dst_dev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                        dstDevice);

  if (src_dev > 0 && dst_dev > 0) {
    LOGI("PeerAsync srcdev:%d dst:%d", src_dev, dst_dev);
    return __CF("cuMemcpyDtoDAsync_v2")(dstDevice, srcDevice, ByteCount,
                                        hStream);
  }

  __C()(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream);
}

CUresult CUDAAPI cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy,
                                     CUstream hStream) {
  __C()(pCopy, hStream);
}

/*

CUresult CUDAAPI cuMemcpyAtoHAsync(void *dstHost, CUarray srcArray, size_t
srcOffset, size_t ByteCount, CUstream hStream)
{
  __C()(dstHost, srcArray, srcOffset, ByteCount, hStream);
}
CUresult CUDAAPI cuMemcpyHtoAAsync(CUarray dstArray, size_t dstOffset, const
void *srcHost, size_t ByteCount, CUstream hStream)
{
  __C()(dstArray, dstOffset, srcHost, ByteCount, hStream);
}
CUresult CUDAAPI cuMemcpy2D(const CUDA_MEMCPY2D *pCopy)
{
  __C()(pCopy);
}

CUresult CUDAAPI cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *pCopy)
{
  __C()(pCopy);
}
CUresult CUDAAPI cuMemcpy3D(const CUDA_MEMCPY3D *pCopy)
{
  __C()(pCopy);
}

CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost,
size_t ByteCount, CUstream hStream)
{
  __C()(dstDevice, srcHost, ByteCount, hStream);
}

CUresult CUDAAPI cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t
ByteCount, CUstream hStream)
{
  __C()(dstHost, srcDevice, ByteCount, hStream);
}

CUresult CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
size_t ByteCount, CUstream hStream)
{
  __C()(dstDevice, srcDevice, ByteCount, hStream);
}

CUresult CUDAAPI cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy, CUstream hStream)
{
  __C()(pCopy, hStream);
}

CUresult CUDAAPI cuMemcpy3DAsync(const CUDA_MEMCPY3D *pCopy, CUstream hStream)
{
  __C()(pCopy, hStream);
}

CUresult CUDAAPI cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N)
{
  __C()(dstDevice, uc, N);
}

CUresult CUDAAPI cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N)
{
  __C()(dstDevice, us, N);
}

CUresult CUDAAPI cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N)
{
  __C()(dstDevice, ui, N);
}

CUresult CUDAAPI cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch, unsigned
char uc, size_t Width, size_t Height)
{
  __C()(dstDevice, dstPitch, uc, Width, Height);
}

CUresult CUDAAPI cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch, unsigned
short us, size_t Width, size_t Height)
{
  __C()(dstDevice, dstPitch, us, Width, Height);
}

CUresult CUDAAPI cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch, unsigned
int ui, size_t Width, size_t Height)
{
  __C()(dstDevice, dstPitch, ui, Width, Height);
}
*/

CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc,
                                 size_t N, CUstream hStream) {
  __C()(dstDevice, uc, N, hStream);
}

CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us,
                                  size_t N, CUstream hStream) {
  __C()(dstDevice, us, N, hStream);
}

CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui,
                                  size_t N, CUstream hStream) {
  __C()(dstDevice, ui, N, hStream);
}

CUresult CUDAAPI cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch,
                                   unsigned char uc, size_t Width,
                                   size_t Height, CUstream hStream) {
  __C()(dstDevice, dstPitch, uc, Width, Height, hStream);
}

CUresult CUDAAPI cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch,
                                    unsigned short us, size_t Width,
                                    size_t Height, CUstream hStream) {
  __C()(dstDevice, dstPitch, us, Width, Height, hStream);
}

CUresult CUDAAPI cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch,
                                    unsigned int ui, size_t Width,
                                    size_t Height, CUstream hStream) {
  __C()(dstDevice, dstPitch, ui, Width, Height, hStream);
}

CUresult CUDAAPI cuArrayCreate(CUarray *pHandle,
                               const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  __C()(pHandle, pAllocateArray);
}

CUresult CUDAAPI cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor,
                                      CUarray hArray) {
  __C()(pArrayDescriptor, hArray);
}

CUresult CUDAAPI cuArrayGetSparseProperties(
    CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties, CUarray array) {
  __C()(sparseProperties, array);
}

CUresult CUDAAPI cuMipmappedArrayGetSparseProperties(
    CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties, CUmipmappedArray mipmap) {
  __C()(sparseProperties, mipmap);
}

CUresult CUDAAPI
cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS *memoryRequirements,
                             CUarray array, CUdevice device) {
  __C()(memoryRequirements, array, device);
}

CUresult CUDAAPI cuMipmappedArrayGetMemoryRequirements(
    CUDA_ARRAY_MEMORY_REQUIREMENTS *memoryRequirements, CUmipmappedArray mipmap,
    CUdevice device) {
  __C()(memoryRequirements, mipmap, device);
}

CUresult CUDAAPI cuArrayGetPlane(CUarray *pPlaneArray, CUarray hArray,
                                 unsigned int planeIdx) {
  __C()(pPlaneArray, hArray, planeIdx);
}

CUresult CUDAAPI cuArrayDestroy(CUarray hArray) { __C()(hArray); }

CUresult CUDAAPI cuArray3DCreate(
    CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  __C()(pHandle, pAllocateArray);
}

CUresult CUDAAPI cuArray3DGetDescriptor(
    CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
  __C()(pArrayDescriptor, hArray);
}

CUresult CUDAAPI
cuMipmappedArrayCreate(CUmipmappedArray *pHandle,
                       const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
                       unsigned int numMipmapLevels) {
  __C()(pHandle, pMipmappedArrayDesc, numMipmapLevels);
}

CUresult CUDAAPI cuMipmappedArrayGetLevel(CUarray *pLevelArray,
                                          CUmipmappedArray hMipmappedArray,
                                          unsigned int level) {
  __C()(pLevelArray, hMipmappedArray, level);
}

CUresult CUDAAPI cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
  __C()(hMipmappedArray);
}

CUresult CUDAAPI cuMemAddressReserve(CUdeviceptr *ptr, size_t size,
                                     size_t alignment, CUdeviceptr addr,
                                     unsigned long long flags) {
  __C()(ptr, size, alignment, addr, flags);
}

CUresult CUDAAPI cuMemAddressFree(CUdeviceptr ptr, size_t size) {
  __C()(ptr, size);
}

CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                             const CUmemAllocationProp *prop,
                             unsigned long long flags) {
  __C()(handle, size, prop, flags);
}

CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle handle) {
  __C()(handle);
}

CUresult CUDAAPI cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                          CUmemGenericAllocationHandle handle,
                          unsigned long long flags) {
  __C()(ptr, size, offset, handle, flags);
}

CUresult CUDAAPI cuMemMapArrayAsync(CUarrayMapInfo *mapInfoList,
                                    unsigned int count, CUstream hStream) {
  __C()(mapInfoList, count, hStream);
}

CUresult CUDAAPI cuMemUnmap(CUdeviceptr ptr, size_t size) { __C()(ptr, size); }

CUresult CUDAAPI cuMemSetAccess(CUdeviceptr ptr, size_t size,
                                const CUmemAccessDesc *desc, size_t count) {
  __C()(ptr, size, desc, count);
}

CUresult CUDAAPI cuMemGetAccess(unsigned long long *flags,
                                const CUmemLocation *location,
                                CUdeviceptr ptr) {
  __C()(flags, location, ptr);
}

CUresult CUDAAPI cuMemExportToShareableHandle(
    void *shareableHandle, CUmemGenericAllocationHandle handle,
    CUmemAllocationHandleType handleType, unsigned long long flags) {
  __C()(shareableHandle, handle, handleType, flags);
}

CUresult CUDAAPI cuMemImportFromShareableHandle(
    CUmemGenericAllocationHandle *handle, void *osHandle,
    CUmemAllocationHandleType shHandleType) {
  __C()(handle, osHandle, shHandleType);
}

CUresult CUDAAPI cuMemGetAllocationGranularity(
    size_t *granularity, const CUmemAllocationProp *prop,
    CUmemAllocationGranularity_flags option) {
  __C()(granularity, prop, option);
}

CUresult CUDAAPI cuMemGetAllocationPropertiesFromHandle(
    CUmemAllocationProp *prop, CUmemGenericAllocationHandle handle) {
  __C()(prop, handle);
}

CUresult CUDAAPI
cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle, void *addr) {
  __C()(handle, addr);
}

// TODO
CUresult CUDAAPI cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
  __C()(dptr, hStream);
}

CUresult CUDAAPI cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize,
                                 CUstream hStream) {
  __C()(dptr, bytesize, hStream);
}

CUresult CUDAAPI cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) {
  __C()(pool, minBytesToKeep);
}

CUresult CUDAAPI cuMemPoolSetAttribute(CUmemoryPool pool,
                                       CUmemPool_attribute attr, void *value) {
  __C()(pool, attr, value);
}

CUresult CUDAAPI cuMemPoolGetAttribute(CUmemoryPool pool,
                                       CUmemPool_attribute attr, void *value) {
  __C()(pool, attr, value);
}

CUresult CUDAAPI cuMemPoolSetAccess(CUmemoryPool pool,
                                    const CUmemAccessDesc *map, size_t count) {
  __C()(pool, map, count);
}

CUresult CUDAAPI cuMemPoolGetAccess(CUmemAccess_flags *flags,
                                    CUmemoryPool memPool,
                                    CUmemLocation *location) {
  __C()(flags, memPool, location);
}

CUresult CUDAAPI cuMemPoolCreate(CUmemoryPool *pool,
                                 const CUmemPoolProps *poolProps) {
  __C()(pool, poolProps);
}

CUresult CUDAAPI cuMemPoolDestroy(CUmemoryPool pool) { __C()(pool); }

CUresult CUDAAPI cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize,
                                         CUmemoryPool pool, CUstream hStream) {
  __C()(dptr, bytesize, pool, hStream);
}

CUresult CUDAAPI cuMemPoolExportToShareableHandle(
    void *handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType,
    unsigned long long flags) {
  __C()(handle_out, pool, handleType, flags);
}

CUresult CUDAAPI cuMemPoolImportFromShareableHandle(
    CUmemoryPool *pool_out, void *handle, CUmemAllocationHandleType handleType,
    unsigned long long flags) {
  __C()(pool_out, handle, handleType, flags);
}

CUresult CUDAAPI cuMemPoolExportPointer(CUmemPoolPtrExportData *shareData_out,
                                        CUdeviceptr ptr) {
  __C()(shareData_out, ptr);
}

CUresult CUDAAPI cuMemPoolImportPointer(CUdeviceptr *ptr_out, CUmemoryPool pool,
                                        CUmemPoolPtrExportData *shareData) {
  __C()(ptr_out, pool, shareData);
}

CUresult CUDAAPI cuPointerGetAttribute(void *data,
                                       CUpointer_attribute attribute,
                                       CUdeviceptr ptr) {
  //__C()(data, attribute, ptr);
  CUresult rst = __CF("cuPointerGetAttribute")(data, attribute, ptr);
  if (rst == CUDA_SUCCESS && attribute == CU_POINTER_ATTRIBUTE_MEMORY_TYPE &&
      g_cfg->get_memMode() == GMM_MEM_MODE_UM) {
    LOGI("================== ptr:%llx type:%d\n", ptr, *(int *)data);
    fflush(stdout);
  }
  return rst;
}

CUresult CUDAAPI cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count,
                                    CUdevice dstDevice, CUstream hStream) {
  __C()(devPtr, count, dstDevice, hStream);
}

CUresult CUDAAPI cuMemAdvise(CUdeviceptr devPtr, size_t count,
                             CUmem_advise advice, CUdevice device) {
  __C()(devPtr, count, advice, device);
}

CUresult CUDAAPI cuMemRangeGetAttribute(void *data, size_t dataSize,
                                        CUmem_range_attribute attribute,
                                        CUdeviceptr devPtr, size_t count) {
  __C()(data, dataSize, attribute, devPtr, count);
}

CUresult CUDAAPI cuMemRangeGetAttributes(void **data, size_t *dataSizes,
                                         CUmem_range_attribute *attributes,
                                         size_t numAttributes,
                                         CUdeviceptr devPtr, size_t count) {
  __C()(data, dataSizes, attributes, numAttributes, devPtr, count);
}

CUresult CUDAAPI cuPointerSetAttribute(const void *value,
                                       CUpointer_attribute attribute,
                                       CUdeviceptr ptr) {
  __C()(value, attribute, ptr);
}

CUresult CUDAAPI cuPointerGetAttributes(unsigned int numAttributes,
                                        CUpointer_attribute *attributes,
                                        void **data, CUdeviceptr ptr) {
  __C()(numAttributes, attributes, data, ptr);
}

CUresult CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int Flags) {
  __C()(phStream, Flags);
}

CUresult CUDAAPI cuStreamCreateWithPriority(CUstream *phStream,
                                            unsigned int flags, int priority) {
  __C()(phStream, flags, priority);
}

CUresult CUDAAPI cuStreamGetPriority(CUstream hStream, int *priority) {
  __C()(hStream, priority);
}

CUresult CUDAAPI cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
  __C()(hStream, flags);
}

CUresult CUDAAPI cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
  __C()(hStream, pctx);
}

CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                                   unsigned int Flags) {
  __C()(hStream, hEvent, Flags);
}

CUresult CUDAAPI cuStreamAddCallback(CUstream hStream,
                                     CUstreamCallback callback, void *userData,
                                     unsigned int flags) {
  __C()(hStream, callback, userData, flags);
}

CUresult CUDAAPI cuStreamBeginCapture_ptsz(CUstream hStream) { __C()(hStream); }

CUresult cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode) {
  __C()(hStream, mode);
}

CUresult CUDAAPI cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode *mode) {
  __C()(mode);
}

CUresult CUDAAPI cuStreamEndCapture(CUstream hStream, CUgraph *phGraph) {
  __C()(hStream, phGraph);
}

CUresult CUDAAPI cuStreamIsCapturing(CUstream hStream,
                                     CUstreamCaptureStatus *captureStatus) {
  __C()(hStream, captureStatus);
}

CUresult CUDAAPI cuStreamGetCaptureInfo(
    CUstream hStream, CUstreamCaptureStatus *captureStatus_out,
    cuuint64_t *id_out) {
  __C()(hStream, captureStatus_out, id_out);
}

CUresult CUDAAPI cuStreamGetCaptureInfo_v2(
    CUstream hStream, CUstreamCaptureStatus *captureStatus_out,
    cuuint64_t *id_out, CUgraph *graph_out,
    const CUgraphNode **dependencies_out, size_t *numDependencies_out) {
  __C()
  (hStream, captureStatus_out, id_out, graph_out, dependencies_out,
   numDependencies_out);
}

CUresult CUDAAPI cuStreamUpdateCaptureDependencies(CUstream hStream,
                                                   CUgraphNode *dependencies,
                                                   size_t numDependencies,
                                                   unsigned int flags) {
  __C()(hStream, dependencies, numDependencies, flags);
}

CUresult CUDAAPI cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr,
                                        size_t length, unsigned int flags) {
  __C()(hStream, dptr, length, flags);
}

CUresult CUDAAPI cuStreamQuery(CUstream hStream) { __C()(hStream); }

CUresult CUDAAPI cuStreamSynchronize(CUstream hStream) {
  CUresult ret = __CF("cuStreamSynchronize")(hStream);

  if (gmm_ctx_p) {  // TODO: should skip GMM worker thread
    gmm_ctx_p->reclaim_stream(hStream, -1);
  }

  return ret;
}

CUresult CUDAAPI cuStreamDestroy(CUstream hStream) {
  if (gmm_ctx_p) {
    gmm_ctx_p->reclaim_stream(hStream, -1);
  }

  __C()(hStream);
}

CUresult CUDAAPI cuStreamCopyAttributes(CUstream dst, CUstream src) {
  __C()(dst, src);
}

CUresult CUDAAPI cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr,
                                      CUstreamAttrValue *value_out) {
  __C()(hStream, attr, value_out);
}

CUresult CUDAAPI cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr,
                                      const CUstreamAttrValue *value) {
  __C()(hStream, attr, value);
}

CUresult CUDAAPI cuEventCreate(CUevent *phEvent, unsigned int Flags) {
  __C()(phEvent, Flags);
}

CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream) {
  if (gmm_ctx_p) gmm_ctx_p->mark_evt(hEvent, hStream);

  __C()(hEvent, hStream);
}

CUresult CUDAAPI cuEventRecordWithFlags(CUevent hEvent, CUstream hStream,
                                        unsigned int flags) {
  if (gmm_ctx_p) gmm_ctx_p->mark_evt(hEvent, hStream);

  __C()(hEvent, hStream, flags);
}

CUresult CUDAAPI cuEventQuery(CUevent hEvent) {
  return __CF("cuEventQuery")(hEvent);
}

CUresult CUDAAPI cuEventSynchronize(CUevent hEvent) {
  // client to sync the evt, first do sync user input hEvent (internally sync
  // any dependent evt) then notify admin to reclaim
  CUresult ret = __CF("cuEventSynchronize")(hEvent);

  if (gmm_ctx_p) {
    gmm_ctx_p->reclaim_evt(hEvent);
  }
  return ret;
}

CUresult CUDAAPI cuEventDestroy(CUevent hEvent) { __C()(hEvent); }

CUresult CUDAAPI cuEventElapsedTime(float *pMilliseconds, CUevent hStart,
                                    CUevent hEnd) {
  __C()(pMilliseconds, hStart, hEnd);
}

CUresult CUDAAPI
cuImportExternalMemory(CUexternalMemory *extMem_out,
                       const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc) {
  __C()(extMem_out, memHandleDesc);
}

CUresult CUDAAPI cuExternalMemoryGetMappedBuffer(
    CUdeviceptr *devPtr, CUexternalMemory extMem,
    const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *bufferDesc) {
  __C()(devPtr, extMem, bufferDesc);
}

CUresult CUDAAPI cuExternalMemoryGetMappedMipmappedArray(
    CUmipmappedArray *mipmap, CUexternalMemory extMem,
    const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc) {
  __C()(mipmap, extMem, mipmapDesc);
}

CUresult CUDAAPI cuDestroyExternalMemory(CUexternalMemory extMem) {
  __C()(extMem);
}

CUresult CUDAAPI cuImportExternalSemaphore(
    CUexternalSemaphore *extSem_out,
    const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc) {
  __C()(extSem_out, semHandleDesc);
}

CUresult CUDAAPI cuSignalExternalSemaphoresAsync(
    const CUexternalSemaphore *extSemArray,
    const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray,
    unsigned int numExtSems, CUstream stream) {
  __C()(extSemArray, paramsArray, numExtSems, stream);
}

CUresult CUDAAPI cuWaitExternalSemaphoresAsync(
    const CUexternalSemaphore *extSemArray,
    const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray,
    unsigned int numExtSems, CUstream stream) {
  __C()(extSemArray, paramsArray, numExtSems, stream);
}

CUresult CUDAAPI cuDestroyExternalSemaphore(CUexternalSemaphore extSem) {
  __C()(extSem);
}

CUresult CUDAAPI cuStreamWaitValue32(CUstream stream, CUdeviceptr addr,
                                     cuuint32_t value, unsigned int flags) {
  __C()(stream, addr, value, flags);
}

CUresult CUDAAPI cuStreamWaitValue64(CUstream stream, CUdeviceptr addr,
                                     cuuint64_t value, unsigned int flags) {
  __C()(stream, addr, value, flags);
}

CUresult CUDAAPI cuStreamWriteValue32(CUstream stream, CUdeviceptr addr,
                                      cuuint32_t value, unsigned int flags) {
  __C()(stream, addr, value, flags);
}

CUresult CUDAAPI cuStreamWriteValue64(CUstream stream, CUdeviceptr addr,
                                      cuuint64_t value, unsigned int flags) {
  __C()(stream, addr, value, flags);
}

CUresult CUDAAPI cuStreamBatchMemOp(CUstream stream, unsigned int count,
                                    CUstreamBatchMemOpParams *paramArray,
                                    unsigned int flags) {
  __C()(stream, count, paramArray, flags);
}

CUresult CUDAAPI cuFuncGetAttribute(int *pi, CUfunction_attribute attrib,
                                    CUfunction hfunc) {
  __C()(pi, attrib, hfunc);
}

CUresult CUDAAPI cuFuncSetAttribute(CUfunction hfunc,
                                    CUfunction_attribute attrib, int value) {
  __C()(hfunc, attrib, value);
}

CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
  __C()(hfunc, config);
}

CUresult CUDAAPI cuFuncSetSharedMemConfig(CUfunction hfunc,
                                          CUsharedconfig config) {
  __C()(hfunc, config);
}

CUresult CUDAAPI cuFuncGetModule(CUmodule *hmod, CUfunction hfunc) {
  __C()(hmod, hfunc);
}

CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                                unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream,
                                void **kernelParams, void **extra) {
  __C()
  (f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
   sharedMemBytes, hStream, kernelParams, extra);
}

CUresult CUDAAPI cuLaunchHostFunc(CUstream hStream, CUhostFn fn,
                                  void *userData) {
  __C()(hStream, fn, userData);
}

CUresult CUDAAPI cuGraphicsMapResources(unsigned int count,
                                        CUgraphicsResource *resources,
                                        CUstream hStream) {
  __C()(count, resources, hStream);
}

CUresult CUDAAPI cuLaunchCooperativeKernel(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams) {
  __CF(__FUNCTION__)
  (f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
   sharedMemBytes, hStream, kernelParams);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLaunchCooperativeKernelMultiDevice(
    CUDA_LAUNCH_PARAMS *launchParamsList, unsigned int numDevices,
    unsigned int flags) {
  __C()(launchParamsList, numDevices, flags);
}

CUresult CUDAAPI cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) {
  __C()(hfunc, x, y, z);
}

CUresult CUDAAPI cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes) {
  __C()(hfunc, bytes);
}

CUresult CUDAAPI cuParamSetSize(CUfunction hfunc, unsigned int numbytes) {
  __C()(hfunc, numbytes);
}

CUresult CUDAAPI cuParamSeti(CUfunction hfunc, int offset, unsigned int value) {
  __C()(hfunc, offset, value);
}

CUresult CUDAAPI cuParamSetf(CUfunction hfunc, int offset, float value) {
  __C()(hfunc, offset);
}

CUresult CUDAAPI cuParamSetv(CUfunction hfunc, int offset, void *ptr,
                             unsigned int numbytes) {
  __C()(hfunc, offset, ptr, numbytes);
}

CUresult CUDAAPI cuLaunch(CUfunction f) { __C()(f); }

CUresult CUDAAPI cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
  __C()(f, grid_width, grid_height);
}

CUresult CUDAAPI cuLaunchGridAsync(CUfunction f, int grid_width,
                                   int grid_height, CUstream hStream) {
  __C()(f, grid_width, grid_height, hStream);
}

CUresult CUDAAPI cuParamSetTexRef(CUfunction hfunc, int texunit,
                                  CUtexref hTexRef) {
  __C()(hfunc, texunit, hTexRef);
}

CUresult CUDAAPI cuGraphCreate(CUgraph *phGraph, unsigned int flags) {
  __C()(phGraph, flags);
}

CUresult CUDAAPI cuGraphAddKernelNode(
    CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
    size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  __C()(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
}

CUresult CUDAAPI cuGraphKernelNodeGetParams(
    CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  __C()(hNode, nodeParams);
}

CUresult CUDAAPI cuGraphKernelNodeSetParams(
    CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  __C()(hNode, nodeParams);
}

CUresult CUDAAPI cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                      const CUgraphNode *dependencies,
                                      size_t numDependencies,
                                      const CUDA_MEMCPY3D *copyParams,
                                      CUcontext ctx) {
  __C()(phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx);
}

CUresult CUDAAPI cuGraphMemcpyNodeGetParams(CUgraphNode hNode,
                                            CUDA_MEMCPY3D *nodeParams) {
  __C()(hNode, nodeParams);
}

CUresult CUDAAPI cuGraphMemcpyNodeSetParams(CUgraphNode hNode,
                                            const CUDA_MEMCPY3D *nodeParams) {
  __C()(hNode, nodeParams);
}

CUresult CUDAAPI cuGraphAddMemsetNode(
    CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
    size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS *memsetParams,
    CUcontext ctx) {
  __C()(phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx);
}

CUresult CUDAAPI cuGraphMemsetNodeGetParams(
    CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS *nodeParams) {
  __C()(hNode, nodeParams);
}

CUresult CUDAAPI cuGraphMemsetNodeSetParams(
    CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *nodeParams) {
  __C()(hNode, nodeParams);
}

CUresult CUDAAPI cuGraphAddHostNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                    const CUgraphNode *dependencies,
                                    size_t numDependencies,
                                    const CUDA_HOST_NODE_PARAMS *nodeParams) {
  __C()(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
}

CUresult CUDAAPI cuGraphHostNodeGetParams(CUgraphNode hNode,
                                          CUDA_HOST_NODE_PARAMS *nodeParams) {
  __C()(hNode, nodeParams);
}

CUresult CUDAAPI cuGraphHostNodeSetParams(
    CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS *nodeParams) {
  __C()(hNode, nodeParams);
}

CUresult CUDAAPI cuGraphAddChildGraphNode(CUgraphNode *phGraphNode,
                                          CUgraph hGraph,
                                          const CUgraphNode *dependencies,
                                          size_t numDependencies,
                                          CUgraph childGraph) {
  __C()(phGraphNode, hGraph, dependencies, numDependencies, childGraph);
}

CUresult CUDAAPI cuGraphChildGraphNodeGetGraph(CUgraphNode hNode,
                                               CUgraph *phGraph) {
  __C()(hNode, phGraph);
}

CUresult CUDAAPI cuGraphAddEmptyNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                     const CUgraphNode *dependencies,
                                     size_t numDependencies) {
  __C()(phGraphNode, hGraph, dependencies, numDependencies);
}

CUresult CUDAAPI cuGraphAddEventRecordNode(CUgraphNode *phGraphNode,
                                           CUgraph hGraph,
                                           const CUgraphNode *dependencies,
                                           size_t numDependencies,
                                           CUevent event) {
  __C()(phGraphNode, hGraph, dependencies, numDependencies, event);
}

CUresult CUDAAPI cuGraphEventRecordNodeGetEvent(CUgraphNode hNode,
                                                CUevent *event_out) {
  __C()(hNode, event_out);
}

CUresult CUDAAPI cuGraphEventRecordNodeSetEvent(CUgraphNode hNode,
                                                CUevent event) {
  __C()(hNode, event);
}

CUresult CUDAAPI cuGraphAddEventWaitNode(CUgraphNode *phGraphNode,
                                         CUgraph hGraph,
                                         const CUgraphNode *dependencies,
                                         size_t numDependencies,
                                         CUevent event) {
  __C()(phGraphNode, hGraph, dependencies, numDependencies, event);
}

CUresult CUDAAPI cuGraphEventWaitNodeGetEvent(CUgraphNode hNode,
                                              CUevent *event_out) {
  __C()(hNode, event_out);
}

CUresult CUDAAPI cuGraphEventWaitNodeSetEvent(CUgraphNode hNode,
                                              CUevent event) {
  __C()(hNode, event);
}

CUresult CUDAAPI cuGraphAddExternalSemaphoresSignalNode(
    CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
    size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
  __C()(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
}

CUresult CUDAAPI cuGraphExternalSemaphoresSignalNodeGetParams(
    CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *params_out) {
  __C()(hNode, params_out);
}

CUresult CUDAAPI cuGraphExternalSemaphoresSignalNodeSetParams(
    CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
  __C()(hNode, nodeParams);
}

CUresult CUDAAPI cuGraphAddExternalSemaphoresWaitNode(
    CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
    size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
  __C()(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
}

CUresult CUDAAPI cuGraphExternalSemaphoresWaitNodeGetParams(
    CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS *params_out) {
  __C()(hNode, params_out);
}

CUresult CUDAAPI cuGraphExternalSemaphoresWaitNodeSetParams(
    CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
  __C()(hNode, nodeParams);
}

CUresult CUDAAPI cuGraphAddMemAllocNode(
    CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
    size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS *nodeParams) {
  __C()(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
}

CUresult CUDAAPI cuGraphMemAllocNodeGetParams(
    CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS *params_out) {
  __C()(hNode, params_out);
}

CUresult CUDAAPI cuGraphAddMemFreeNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                       const CUgraphNode *dependencies,
                                       size_t numDependencies,
                                       CUdeviceptr dptr) {
  __C()(phGraphNode, hGraph, dependencies, numDependencies, dptr);
}

CUresult CUDAAPI cuGraphMemFreeNodeGetParams(CUgraphNode hNode,
                                             CUdeviceptr *dptr_out) {
  __C()(hNode, dptr_out);
}

CUresult CUDAAPI cuDeviceGraphMemTrim(CUdevice device) { __C()(device); }

CUresult CUDAAPI cuDeviceGetGraphMemAttribute(CUdevice device,
                                              CUgraphMem_attribute attr,
                                              void *value) {
  __C()(device, attr, value);
}

CUresult CUDAAPI cuDeviceSetGraphMemAttribute(CUdevice device,
                                              CUgraphMem_attribute attr,
                                              void *value) {
  __C()(device, attr, value);
}

CUresult CUDAAPI cuGraphClone(CUgraph *phGraphClone, CUgraph originalGraph) {
  __C()(phGraphClone, originalGraph);
}

CUresult CUDAAPI cuGraphNodeFindInClone(CUgraphNode *phNode,
                                        CUgraphNode hOriginalNode,
                                        CUgraph hClonedGraph) {
  __C()(phNode, hOriginalNode, hClonedGraph);
}

CUresult CUDAAPI cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType *type) {
  __C()(hNode, type);
}

CUresult CUDAAPI cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes,
                                 size_t *numNodes) {
  __C()(hGraph, nodes, numNodes);
}

CUresult CUDAAPI cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode *rootNodes,
                                     size_t *numRootNodes) {
  __C()(hGraph, rootNodes, numRootNodes);
}

CUresult CUDAAPI cuGraphGetEdges(CUgraph hGraph, CUgraphNode *from,
                                 CUgraphNode *to, size_t *numEdges) {
  __C()(hGraph, from, to, numEdges);
}

CUresult CUDAAPI cuGraphNodeGetDependencies(CUgraphNode hNode,
                                            CUgraphNode *dependencies,
                                            size_t *numDependencies) {
  __C()(hNode, dependencies, numDependencies);
}

CUresult CUDAAPI cuGraphNodeGetDependentNodes(CUgraphNode hNode,
                                              CUgraphNode *dependentNodes,
                                              size_t *numDependentNodes) {
  __C()(hNode, dependentNodes, numDependentNodes);
}

CUresult CUDAAPI cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode *from,
                                        const CUgraphNode *to,
                                        size_t numDependencies) {
  __C()(hGraph, from, to, numDependencies);
}

CUresult CUDAAPI cuGraphRemoveDependencies(CUgraph hGraph,
                                           const CUgraphNode *from,
                                           const CUgraphNode *to,
                                           size_t numDependencies) {
  __C()(hGraph, from, to, numDependencies);
}

CUresult CUDAAPI cuGraphDestroyNode(CUgraphNode hNode) { __C()(hNode); }

CUresult CUDAAPI cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph,
                                    CUgraphNode *phErrorNode, char *logBuffer,
                                    size_t bufferSize) {
  __C()(phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize);
}

CUresult CUDAAPI cuGraphInstantiateWithFlags(CUgraphExec *phGraphExec,
                                             CUgraph hGraph,
                                             unsigned long long flags) {
  __C()(phGraphExec, hGraph, flags);
}

CUresult CUDAAPI
cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                               const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  __C()(hGraphExec, hNode, nodeParams);
}

CUresult CUDAAPI cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec,
                                                CUgraphNode hNode,
                                                const CUDA_MEMCPY3D *copyParams,
                                                CUcontext ctx) {
  __C()(hGraphExec, hNode, copyParams, ctx);
}

CUresult CUDAAPI cuGraphExecMemsetNodeSetParams(
    CUgraphExec hGraphExec, CUgraphNode hNode,
    const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx) {
  __C()(hGraphExec, hNode, memsetParams, ctx);
}

CUresult CUDAAPI
cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                             const CUDA_HOST_NODE_PARAMS *nodeParams) {
  __C()(hGraphExec, hNode, nodeParams);
}

CUresult CUDAAPI cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec,
                                                    CUgraphNode hNode,
                                                    CUgraph childGraph) {
  __C()(hGraphExec, hNode, childGraph);
}

CUresult CUDAAPI cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec,
                                                    CUgraphNode hNode,
                                                    CUevent event) {
  __C()(hGraphExec, hNode, event);
}

CUresult CUDAAPI cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec,
                                                  CUgraphNode hNode,
                                                  CUevent event) {
  __C()(hGraphExec, hNode, event);
}

CUresult CUDAAPI cuGraphExecExternalSemaphoresSignalNodeSetParams(
    CUgraphExec hGraphExec, CUgraphNode hNode,
    const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
  __C()(hGraphExec, hNode, nodeParams);
}

CUresult CUDAAPI cuGraphExecExternalSemaphoresWaitNodeSetParams(
    CUgraphExec hGraphExec, CUgraphNode hNode,
    const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
  __C()(hGraphExec, hNode, nodeParams);
}

CUresult CUDAAPI cuGraphNodeSetEnabled(CUgraphExec hGraphExec,
                                       CUgraphNode hNode,
                                       unsigned int isEnabled) {
  __C()(hGraphExec, hNode, isEnabled);
}

CUresult CUDAAPI cuGraphNodeGetEnabled(CUgraphExec hGraphExec,
                                       CUgraphNode hNode,
                                       unsigned int *isEnabled) {
  __C()(hGraphExec, hNode, isEnabled);
}

CUresult CUDAAPI cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) {
  __C()(hGraphExec, hStream);
}

CUresult CUDAAPI cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
  __C()(hGraphExec, hStream);
}

CUresult CUDAAPI cuGraphExecDestroy(CUgraphExec hGraphExec) {
  __C()(hGraphExec);
}

CUresult CUDAAPI cuGraphDestroy(CUgraph hGraph) { __C()(hGraph); }

CUresult CUDAAPI cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph,
                                   CUgraphNode *hErrorNode_out,
                                   CUgraphExecUpdateResult *updateResult_out) {
  __C()(hGraphExec, hGraph, hErrorNode_out, updateResult_out);
}

CUresult CUDAAPI cuGraphKernelNodeCopyAttributes(CUgraphNode dst,
                                                 CUgraphNode src) {
  __C()(dst, src);
}

CUresult CUDAAPI
cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr,
                              CUkernelNodeAttrValue *value_out) {
  __C()(hNode, attr, value_out);
}

CUresult CUDAAPI
cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr,
                              const CUkernelNodeAttrValue *value) {
  __C()(hNode, attr, value);
}

CUresult CUDAAPI cuGraphDebugDotPrint(CUgraph hGraph, const char *path,
                                      unsigned int flags) {
  __C()(hGraph, path, flags);
}

CUresult CUDAAPI cuUserObjectCreate(CUuserObject *object_out, void *ptr,
                                    CUhostFn destroy,
                                    unsigned int initialRefcount,
                                    unsigned int flags) {
  __C()(object_out, ptr, destroy, initialRefcount, flags);
}

CUresult CUDAAPI cuUserObjectRetain(CUuserObject object, unsigned int count) {
  __C()(object, count);
}

CUresult CUDAAPI cuUserObjectRelease(CUuserObject object, unsigned int count) {
  __C()(object, count);
}

CUresult CUDAAPI cuGraphRetainUserObject(CUgraph graph, CUuserObject object,
                                         unsigned int count,
                                         unsigned int flags) {
  __C()(graph, object, count, flags);
}

CUresult CUDAAPI cuGraphReleaseUserObject(CUgraph graph, CUuserObject object,
                                          unsigned int count) {
  __C()(graph, object, count);
}

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
  __C()(numBlocks, func, blockSize, dynamicSMemSize);
}

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
  __C()(numBlocks, func, blockSize, dynamicSMemSize, flags);
}

CUresult CUDAAPI cuOccupancyMaxPotentialBlockSize(
    int *minGridSize, int *blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize,
    int blockSizeLimit) {
  __C()
  (minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize,
   blockSizeLimit);
}

CUresult CUDAAPI cuOccupancyMaxPotentialBlockSizeWithFlags(
    int *minGridSize, int *blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize,
    int blockSizeLimit, unsigned int flags) {
  __C()
  (minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize,
   blockSizeLimit, flags);
}

CUresult CUDAAPI cuOccupancyAvailableDynamicSMemPerBlock(
    size_t *dynamicSmemSize, CUfunction func, int numBlocks, int blockSize) {
  __C()(dynamicSmemSize, func, numBlocks, blockSize);
}

CUresult CUDAAPI cuTexRefSetArray(CUtexref hTexRef, CUarray hArray,
                                  unsigned int Flags) {
  __C()(hTexRef, hArray, Flags);
}

CUresult CUDAAPI cuTexRefSetMipmappedArray(CUtexref hTexRef,
                                           CUmipmappedArray hMipmappedArray,
                                           unsigned int Flags) {
  __C()(hTexRef, hMipmappedArray, Flags);
}

CUresult CUDAAPI cuTexRefSetAddress(size_t *ByteOffset, CUtexref hTexRef,
                                    CUdeviceptr dptr, size_t bytes) {
  __C()(ByteOffset, hTexRef, dptr, bytes);
}

/*
CUresult CUDAAPI cuTexRefSetAddress2D(CUtexref hTexRef, const
CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, size_t Pitch)
{
  __C()(hTexRef, desc, dptr, Pitch);
}
*/

CUresult CUDAAPI cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt,
                                   int NumPackedComponents) {
  __C()(hTexRef, fmt, NumPackedComponents);
}

CUresult CUDAAPI cuTexRefSetAddressMode(CUtexref hTexRef, int dim,
                                        CUaddress_mode am) {
  __C()(hTexRef, dim, am);
}

CUresult CUDAAPI cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
  __C()(hTexRef, fm);
}

CUresult CUDAAPI cuTexRefSetMipmapFilterMode(CUtexref hTexRef,
                                             CUfilter_mode fm) {
  __C()(hTexRef, fm);
}

CUresult CUDAAPI cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) {
  __C()(hTexRef, bias);
}

CUresult CUDAAPI cuTexRefSetMipmapLevelClamp(CUtexref hTexRef,
                                             float minMipmapLevelClamp,
                                             float maxMipmapLevelClamp) {
  __C()(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp);
}

CUresult CUDAAPI cuTexRefSetMaxAnisotropy(CUtexref hTexRef,
                                          unsigned int maxAniso) {
  __C()(hTexRef, maxAniso);
}

CUresult CUDAAPI cuTexRefSetBorderColor(CUtexref hTexRef, float *pBorderColor) {
  __C()(hTexRef, pBorderColor);
}

CUresult CUDAAPI cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
  __C()(hTexRef, Flags);
}

CUresult CUDAAPI cuTexRefGetAddress(CUdeviceptr *pdptr, CUtexref hTexRef) {
  __C()(pdptr, hTexRef);
}

CUresult CUDAAPI cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef) {
  __C()(phArray, hTexRef);
}

CUresult CUDAAPI cuTexRefGetMipmappedArray(CUmipmappedArray *phMipmappedArray,
                                           CUtexref hTexRef) {
  __C()(phMipmappedArray, hTexRef);
}

CUresult CUDAAPI cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef,
                                        int dim) {
  __C()(pam, hTexRef, dim);
}

CUresult CUDAAPI cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
  __C()(pfm, hTexRef);
}

CUresult CUDAAPI cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels,
                                   CUtexref hTexRef) {
  __C()(pFormat, pNumChannels, hTexRef);
}

CUresult CUDAAPI cuTexRefGetMipmapFilterMode(CUfilter_mode *pfm,
                                             CUtexref hTexRef) {
  __C()(pfm, hTexRef);
}

CUresult CUDAAPI cuTexRefGetMipmapLevelBias(float *pbias, CUtexref hTexRef) {
  __C()(pbias, hTexRef);
}

CUresult CUDAAPI cuTexRefGetMipmapLevelClamp(float *pminMipmapLevelClamp,
                                             float *pmaxMipmapLevelClamp,
                                             CUtexref hTexRef) {
  __C()(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef);
}

CUresult CUDAAPI cuTexRefGetMaxAnisotropy(int *pmaxAniso, CUtexref hTexRef) {
  __C()(pmaxAniso, hTexRef);
}

CUresult CUDAAPI cuTexRefGetBorderColor(float *pBorderColor, CUtexref hTexRef) {
  __C()(pBorderColor, hTexRef);
}

CUresult CUDAAPI cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef) {
  __C()(pFlags, hTexRef);
}

CUresult CUDAAPI cuTexRefCreate(CUtexref *pTexRef) { __C()(pTexRef); }

CUresult CUDAAPI cuTexRefDestroy(CUtexref hTexRef) { __C()(hTexRef); }

CUresult CUDAAPI cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray,
                                   unsigned int Flags) {
  __C()(hSurfRef, hArray, Flags);
}

CUresult CUDAAPI cuSurfRefGetArray(CUarray *phArray, CUsurfref hSurfRef) {
  __C()(phArray, hSurfRef);
}

CUresult CUDAAPI
cuTexObjectCreate(CUtexObject *pTexObject, const CUDA_RESOURCE_DESC *pResDesc,
                  const CUDA_TEXTURE_DESC *pTexDesc,
                  const CUDA_RESOURCE_VIEW_DESC *pResViewDesc) {
  __C()(pTexObject, pResDesc, pTexDesc, pResViewDesc);
}

CUresult CUDAAPI cuTexObjectDestroy(CUtexObject texObject) { __C()(texObject); }

CUresult CUDAAPI cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                            CUtexObject texObject) {
  __C()(pResDesc, texObject);
}

CUresult CUDAAPI cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc,
                                           CUtexObject texObject) {
  __C()(pTexDesc, texObject);
}

CUresult CUDAAPI cuTexObjectGetResourceViewDesc(
    CUDA_RESOURCE_VIEW_DESC *pResViewDesc, CUtexObject texObject) {
  __C()(pResViewDesc, texObject);
}

CUresult CUDAAPI cuSurfObjectCreate(CUsurfObject *pSurfObject,
                                    const CUDA_RESOURCE_DESC *pResDesc) {
  __C()(pSurfObject, pResDesc);
}

CUresult CUDAAPI cuSurfObjectDestroy(CUsurfObject surfObject) {
  __C()(surfObject);
}

CUresult CUDAAPI cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                             CUsurfObject surfObject) {
  __C()(pResDesc, surfObject);
}

CUresult CUDAAPI cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev,
                                       CUdevice peerDev) {
  __C()(canAccessPeer, dev, peerDev);
}

CUresult CUDAAPI cuCtxEnablePeerAccess(CUcontext peerContext,
                                       unsigned int Flags) {
  __C()(peerContext, Flags);
}

CUresult CUDAAPI cuCtxDisablePeerAccess(CUcontext peerContext) {
  __C()(peerContext);
}

CUresult CUDAAPI cuDeviceGetP2PAttribute(int *value,
                                         CUdevice_P2PAttribute attrib,
                                         CUdevice srcDevice,
                                         CUdevice dstDevice) {
  __C()(value, attrib, srcDevice, dstDevice);
}

CUresult CUDAAPI cuGraphicsUnregisterResource(CUgraphicsResource resource) {
  __C()(resource);
}

CUresult CUDAAPI cuGraphicsSubResourceGetMappedArray(
    CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex,
    unsigned int mipLevel) {
  __C()(pArray, resource, arrayIndex, mipLevel);
}

CUresult CUDAAPI cuGraphicsResourceGetMappedMipmappedArray(
    CUmipmappedArray *pMipmappedArray, CUgraphicsResource resource) {
  __C()(pMipmappedArray, resource);
}

CUresult CUDAAPI cuGraphicsResourceGetMappedPointer(
    CUdeviceptr *pDevPtr, size_t *pSize, CUgraphicsResource resource) {
  __C()(pDevPtr, pSize, resource);
}

CUresult CUDAAPI cuGraphicsResourceSetMapFlags(CUgraphicsResource resource,
                                               unsigned int flags) {
  __C()(resource, flags);
}

CUresult CUDAAPI cuGraphicsUnmapResources(unsigned int count,
                                          CUgraphicsResource *resources,
                                          CUstream hStream) {
  __C()(count, resources, hStream);
}

// explicitly set pfn IF customization is needed
CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn,
                                  int cudaVersion, cuuint64_t flags) {
  int version_ = cudaVersion;
  size_t len = strlen(symbol);
  LOGD("---->api:%s version_in:%d set:%d flag:%lx pid:%d ppid:%d", symbol,
       cudaVersion, version_, flags, getpid(), getppid());

  if (g_cfg->get_cuda_back()) {
    version_ = 11020;  // test
    __C()(symbol, pfn, version_, flags);
  } else {
    // note: compare shorter name prior to longer name
    if (strncmp(symbol, "cuGetProcAddress", len) == 0) {
      *pfn = (void *)cuGetProcAddress;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuInit", len) == 0) {
      *pfn = (void *)cuInit;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuCtxCreate", len) == 0) {
      *pfn = (void *)cuCtxCreate;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuDevicePrimaryCtxRetain", len) == 0) {
      *pfn = (void *)cuDevicePrimaryCtxRetain;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuCtxGetCurrent", len) == 0) {
      *pfn = (void *)cuCtxGetCurrent;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuCtxSetCurrent", len) == 0) {
      *pfn = (void *)cuCtxSetCurrent;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuCtxGetApiVersion", len) == 0) {
      *pfn = (void *)cuCtxGetApiVersion;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuCtxGetDevice", len) == 0) {
      *pfn = (void *)cuCtxGetDevice;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuMemGetInfo", len) == 0) {
      *pfn = (void *)cuMemGetInfo;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuMemcpyHtoD", len) == 0) {
      *pfn = (void *)cuMemcpyHtoD;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuMemcpyHtoDAsync", len) == 0) {
      *pfn = (void *)cuMemcpyHtoDAsync;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuMemcpyDtoH", len) == 0) {
      *pfn = (void *)cuMemcpyDtoH;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuMemcpyDtoHAsync", len) == 0) {
      *pfn = (void *)cuMemcpyDtoHAsync;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuMemcpyDtoD", len) == 0) {
      *pfn = (void *)cuMemcpyDtoD;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuMemcpyDtoDAsync", len) == 0) {
      *pfn = (void *)cuMemcpyDtoDAsync;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuMemAlloc", len) == 0) {
      *pfn = (void *)cuMemAlloc;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuMemAllocHost", len) == 0) {
      *pfn = (void *)cuMemAllocHost;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuMemHostAlloc", len) == 0) {
      *pfn = (void *)cuMemHostAlloc;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuMemFree", len) == 0) {
      *pfn = (void *)cuMemFree;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuMemFreeHost", len) == 0) {
      *pfn = (void *)cuMemFreeHost;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuEventRecord", len) == 0) {
      *pfn = (void *)cuEventRecord;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuEventRecordWithFlags", len) == 0) {
      *pfn = (void *)cuEventRecordWithFlags;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuEventSynchronize", len) == 0) {
      *pfn = (void *)cuEventSynchronize;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuEventQuery", len) == 0) {
      *pfn = (void *)cuEventQuery;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuEventElapsedTime", len) == 0) {
      *pfn = (void *)cuEventElapsedTime;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuStreamQuery", len) == 0) {
      *pfn = (void *)cuStreamQuery;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuStreamSynchronize", len) == 0) {
      *pfn = (void *)cuStreamSynchronize;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuDeviceGet", len) == 0) {
      *pfn = (void *)cuDeviceGet;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuMemHostRegister", len) == 0) {
      *pfn = (void *)cuMemHostRegister;
      return CUDA_SUCCESS;
    } else if (strncmp(symbol, "cuMemHostUnregister", len) == 0) {
      *pfn = (void *)cuMemHostUnregister;
      return CUDA_SUCCESS;
    } else {
      __C()(symbol, pfn, version_, flags);
    }
  }
}

CUresult CUDAAPI cuGetExportTable(const void **ppExportTable,
                                  const CUuuid *pExportTableId) {
  __C()(ppExportTable, pExportTableId);
}

CUresult CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option *options,
                              void **optionValues, CUlinkState *stateOut) {
  __C()(numOptions, options, optionValues, stateOut);
}

CUresult CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type,
                               void *data, size_t size, const char *name,
                               unsigned int numOptions, CUjit_option *options,
                               void **optionValues) {
  __C()(state, type, data, size, name, numOptions, options, optionValues);
}

CUresult CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type,
                               const char *path, unsigned int numOptions,
                               CUjit_option *options, void **optionValues) {
  __C()(state, type, path, numOptions, options, optionValues);
}

CUresult CUDAAPI cuTexRefSetAddress2D_v2(CUtexref hTexRef,
                                         const CUDA_ARRAY_DESCRIPTOR *desc,
                                         CUdeviceptr dptr, size_t Pitch) {
  __C()(hTexRef, desc, dptr, Pitch);
}

CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost,
                                 size_t ByteCount) {
  if (gmm_ctx_p &&
      (0 == gmm_ctx_p->htod((char *)dstDevice, (char *)srcHost, ByteCount))) {
    return CUDA_SUCCESS;
  }
  __C()(dstDevice, srcHost, ByteCount);
}

CUresult CUDAAPI cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice,
                                 size_t ByteCount) {
  if (gmm_ctx_p && gmm_ctx_p->mp_ok(ByteCount) &&
      (0 == gmm_ctx_p->dtoh((char *)dstHost, (char *)srcDevice, ByteCount))) {
    return CUDA_SUCCESS;
  }
  __C()(dstHost, srcDevice, ByteCount);
}

CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                 size_t ByteCount) {
  if (gmm_ctx_p && gmm_ctx_p->mp_ok(ByteCount) &&
      (gmm_ctx_p->dtod_mp_ok((char *)dstDevice, (char *)srcDevice) >= 0) &&
      (0 == gmm_ctx_p->dtod((char *)dstDevice, (char *)srcDevice, ByteCount))) {
    return CUDA_SUCCESS;
  }
  __C()(dstDevice, srcDevice, ByteCount);
}

CUresult CUDAAPI cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset,
                                 CUdeviceptr srcDevice, size_t ByteCount) {
  __C()(dstArray, dstOffset, srcDevice, ByteCount);
}

CUresult CUDAAPI cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray,
                                 size_t srcOffset, size_t ByteCount) {
  __C()(dstDevice, srcArray, srcOffset, ByteCount);
}

CUresult CUDAAPI cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset,
                                 const void *srcHost, size_t ByteCount) {
  __C()(dstArray, dstOffset, srcHost, ByteCount);
}

CUresult CUDAAPI cuMemcpyAtoH_v2(void *dstHost, CUarray srcArray,
                                 size_t srcOffset, size_t ByteCount) {
  __C()(dstHost, srcArray, srcOffset, ByteCount);
}

CUresult CUDAAPI cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset,
                                 CUarray srcArray, size_t srcOffset,
                                 size_t ByteCount) {
  __C()(dstArray, dstOffset, srcArray, srcOffset, ByteCount);
}

CUresult CUDAAPI cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset,
                                      const void *srcHost, size_t ByteCount,
                                      CUstream hStream) {
  __C()(dstArray, dstOffset, srcHost, ByteCount, hStream);
}

CUresult CUDAAPI cuMemcpyAtoHAsync_v2(void *dstHost, CUarray srcArray,
                                      size_t srcOffset, size_t ByteCount,
                                      CUstream hStream) {
  __C()(dstHost, srcArray, srcOffset, ByteCount, hStream);
}

CUresult CUDAAPI cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy) { __C()(pCopy); }

CUresult CUDAAPI cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy) {
  __C()(pCopy);
}

int glake::SwapOut(void *host_addr, void *dev_addr, size_t bytes,
                   const CUstream &stream) {
  if (gmm_ctx_p && (0 == gmm_ctx_p->scatter((char *)host_addr, (char *)dev_addr,
                                            bytes, stream))) {
    return 0;
  } else {
    return -1;
  }
}

int glake::SwapIn(void *dev_addr, void *host_addr, size_t bytes,
                  const CUstream &stream) {
  if (gmm_ctx_p && (0 == gmm_ctx_p->gather((char *)dev_addr, (char *)host_addr,
                                           bytes, stream))) {
    return 0;
  } else {
    return -1;
  }
}

int glake::H2DMultiPath(void *dev_addr, void *host_addr, size_t bytes,
                        CUstream &stream) {
  if (gmm_ctx_p &&
      (0 == gmm_ctx_p->htod_async((char *)dev_addr, (char *)host_addr, bytes,
                                  stream))) {
    return 0;
  } else {
    printf("%s Fail\n", __func__);
    return -1;
  }
}

int glake::D2HMultiPath(void *host_addr, void *dev_addr, size_t bytes,
                        CUstream &stream) {
  if (gmm_ctx_p &&
      (0 == gmm_ctx_p->dtoh_async((char *)host_addr, (char *)dev_addr, bytes,
                                  stream))) {
    return 0;
  } else {
    printf("%s Fail\n", __func__);
    return -1;
  }
}

CUresult CUDAAPI glake::fetch(void *dst, void *src, size_t ByteCount,
                              CUstream hStream) {
  if (gmm_ctx_p &&
      (0 == gmm_ctx_p->fetch((char *)dst, (char *)src, ByteCount, hStream))) {
    return CUDA_SUCCESS;
  }
  return CUDA_SUCCESS;
}

CUresult CUDAAPI glake::evict(void *srcDevice, size_t ByteCount,
                              CUstream hStream) {
  if (gmm_ctx_p && gmm_ctx_p->mp_ok(ByteCount) &&
      (0 == gmm_ctx_p->evict((char *)srcDevice, ByteCount, hStream))) {
    return CUDA_SUCCESS;
  }
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice,
                                      const void *srcHost, size_t ByteCount,
                                      CUstream hStream) {
  if (IsMultiPath() && gmm_ctx_p &&
      (0 == gmm_ctx_p->htod_async((char *)dstDevice, (char *)srcHost, ByteCount,
                                  hStream))) {
    // printf("MP_OK %s, size:%zu, h_ptr:%p, d_ptr:%p, IsMultiPath:%d\n",
    //		  __func__, ByteCount, (char *)srcHost, (char *)dstDevice,
    //IsMultiPath());
    return CUDA_SUCCESS;
  } else {
    // printf("[cuda] Warning: htod_async fail, use CUDA API.\n");
  }

  __C()(dstDevice, srcHost, ByteCount, hStream);
}

CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                                      size_t ByteCount, CUstream hStream) {
  int ret = 0;
  if (IsMultiPath() && gmm_ctx_p && gmm_ctx_p->mp_ok(ByteCount)) {
    ret = gmm_ctx_p->dtoh_async((char *)dstHost, (char *)srcDevice, ByteCount,
                                hStream);
    if (ret == 0) {
      // printf("MP_OK %s, size:%zu, h_ptr:%p, d_ptr:%p, IsMultiPath:%d\n",
      //		  __func__, ByteCount, (char *)dstHost, (char
      //*)srcDevice, IsMultiPath());
      return CUDA_SUCCESS;
    } else {
      // printf("[cuda] Warning: dtoh_async fail, use CUDA API.\n");
    }
  }

  __C()(dstHost, srcDevice, ByteCount, hStream);
}

CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice,
                                      CUdeviceptr srcDevice, size_t ByteCount,
                                      CUstream hStream) {
  if (gmm_ctx_p && gmm_ctx_p->mp_ok(ByteCount) &&
      (gmm_ctx_p->dtod_mp_ok((char *)dstDevice, (char *)srcDevice) >= 0) &&
      (0 == gmm_ctx_p->dtod_async((char *)dstDevice, (char *)srcDevice,
                                  ByteCount, hStream))) {
    return CUDA_SUCCESS;
  }

  __C()(dstDevice, srcDevice, ByteCount, hStream);
}

CUresult CUDAAPI cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy,
                                    CUstream hStream) {
  __C()(pCopy, hStream);
}

CUresult CUDAAPI cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy,
                                    CUstream hStream) {
  __C()(pCopy, hStream);
}

CUresult CUDAAPI cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc,
                               size_t N) {
  __C()(dstDevice, uc, N);
}

CUresult CUDAAPI cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us,
                                size_t N) {
  __C()(dstDevice, us, N);
}

CUresult CUDAAPI cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui,
                                size_t N) {
  __C()(dstDevice, ui, N);
}

CUresult CUDAAPI cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch,
                                 unsigned char uc, size_t Width,
                                 size_t Height) {
  __C()(dstDevice, dstPitch, uc, Width, Height);
}

CUresult CUDAAPI cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch,
                                  unsigned short us, size_t Width,
                                  size_t Height) {
  __C()(dstDevice, dstPitch, us, Width, Height);
}

CUresult CUDAAPI cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch,
                                  unsigned int ui, size_t Width,
                                  size_t Height) {
  __C()(dstDevice, dstPitch, ui, Width, Height);
}

//////////////////////////
#ifdef __cplusplus
}
#endif
