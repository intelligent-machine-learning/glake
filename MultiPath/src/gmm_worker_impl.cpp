#include <fcntl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>

#include <chrono>
#include <memory>
#include <unordered_map>

#include "gmm_host_shm.h"
#include "gmm_mp.h"
#include "gmm_singleton.h"
#include "gmm_worker.h"

#define MODULE_STATUS CUresult
extern void *libP;

extern std::chrono::time_point<std::chrono::steady_clock> g_req_start_t;

// map from vmm_handle to fd_shared
struct fd_shared_t {
  int client_id;  // connect id
  int shared_fd;  // shareable fd
  pid_t pid;      // src client pid
  size_t bytes;   // alloc bytes
};

static std::unordered_map<CUmemGenericAllocationHandle, fd_shared_t>
    gmm_handle_map;
static void gmm_mem_cleanup(int worker, int connectID, size_t &borrow_sz) {
  for (auto iter = gmm_handle_map.begin(); iter != gmm_handle_map.end();) {
    if (iter->second.client_id == connectID) {
      int fd = iter->second.shared_fd;
      close(fd);
      CHECK_DRV(cuMemRelease(iter->first));
      if (borrow_sz >= iter->second.bytes) borrow_sz -= iter->second.bytes;

      LOGGER(INFO,
             "worker:%d clean up pid:%d fd:%d bytes:%ld cur_borrow:%ld MB",
             worker, iter->second.pid, fd, iter->second.bytes, borrow_sz >> 20);

      auto cur = iter;
      ++iter;
      gmm_handle_map.erase(cur);
    } else {
      ++iter;
    }
  }
}

static bool IsPcieRxNotBusy() { return true; }

static bool IsPcieTxNotBusy() { return true; }

static bool IsNvlinkRxNotBusy() { return true; }

static bool IsNvlinkTxNotBusy() { return true; }

static bool IsLocalGpuMemTight() { return false; }

static uint64_t GetGpuFreeSize() { return 0; }

int32_t TieringCache::Shrink(uint64_t size) {
  // TODO
  return 0;
}

// Find and pop the block with id if complete.
// Return status:
//  0 = Success, 1 = Found but not complete, 2 = Not found.
int32_t TieringCache::PopCompleteBlock(
    std::deque<std::unique_ptr<TieringCache::Block>> &queue, uint64_t id,
    std::unique_ptr<TieringCache::Block> *blk) {
  // std::unique_ptr<TieringCache::Block> blk;
  auto it = queue.rbegin();
  // printf("-- PoPBlock size:%zu\n", queue.size());
  while (it != queue.rend()) {
    if ((*it)->id_ == id) {
      if (this->IsBlockComplete(**it)) {
        *blk = std::move(*it);
        it = decltype(it)(queue.erase(std::next(it).base()));
        // printf("-- %s OK id:%ld\n", __func__, id);
        return 0;
      } else {
        // printf("-- %s NULL due to non-complete id:%ld\n", __func__, id);
        return 1;
      }
    } else {
      // printf("-- PoPBlock Not equal queue id:%ld id:%ld\n", (*it)->id_, id);
      it++;
    }
  }
  // printf("-- %s NULL due to not found id:%ld\n", __func__, id);
  return 2;
}

void TieringCache::PopByHostPtr(
    std::deque<std::unique_ptr<TieringCache::Block>> &queue,
    const char *const host_p, std::unique_ptr<TieringCache::Block> *blk) {
  auto it = queue.begin();
  while (it != queue.end()) {
    if ((*it)->host_ptr == host_p) {
      *blk = std::move(*it);
      it = queue.erase(it);
    } else {
      // printf("-- Search: host_p:%p rm_queue:%p\n", host_p, (*it)->host_ptr);
      it++;
    }
  }
}

// return 0: use block to copy
//        1: direct copy
//      < 0: error
int32_t TieringCache::CopyFromDev(const UnitReq &ureq, CUevent *evt,
                                  uint64_t *id) {
  this->Update();

  // 1. Find a block.
  std::unique_ptr<TieringCache::Block> blk;
  if (this->CanAlloc()) {
    blk = this->NewBlock();
    // 1.1 Allocate from pool if possible.
    int ret =
        pool_->alloc((CUdeviceptr *)(&blk->d_ptr_), kBlockSize_, alloc_stream_);
    if (blk->d_ptr_ == nullptr) {
      printf("Error pool alloc\n");
      abort();
    }
    size_ += kBlockSize_;
    CHECK_DRV(cuEventRecord(alloc_evt_, alloc_stream_));
  } else {
    // 1.2 Reuse a block from rm_queue_.
    if (rm_queue_.empty()) {
      // Direct copy from dev to host.
      gmm_launch_cp_kernel(ureq.dst, ureq.src, ureq.size, ureq.stream, *evt,
                           *evt);
      return 1;
    }
    blk = std::move(rm_queue_.back());
    rm_queue_.pop_back();
  }
  blk->host_ptr = ureq.dst;

  // 2. Copy from dev to the block.
  CHECK_DRV(cuStreamWaitEvent(d2h_1st_stream_, alloc_evt_, 0));
  // printf("--Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
  // CHECK_DRV(cuMemcpyAsync((CUdeviceptr)blk->d_ptr_, (CUdeviceptr)ureq.src,
  // ureq.size, d2h_1st_stream_)); CHECK_CUDA(cudaMemcpyAsync((void*)blk->d_ptr_,
  // (void*)ureq.src, ureq.size, cudaMemcpyDeviceToDevice, d2h_1st_stream_));
  gmm_launch_cp_kernel(blk->d_ptr_, ureq.src, ureq.size, d2h_1st_stream_,
                       blk->evt_, blk->evt_);
  // cudaStreamSynchronize(d2h_1st_stream_);
  CHECK_DRV(cuEventRecord(blk->evt_, d2h_1st_stream_));

  *id = blk->id_;
  *evt = blk->evt_;
  d2h_1st_queue_.push_front(std::move(blk));
  return 0;
}

// Returen status:
//  0: Success
//  1: Not found the blk, as the 1st stage has not completed
// <0: Error
int32_t TieringCache::CopyToHost(const UnitReq &ureq, CUevent *evt,
                                 uint64_t id) {
  this->Update();

  // 1. Find the block with id.
  std::unique_ptr<TieringCache::Block> blk;
  int ret = this->PopCompleteBlock(d2h_1st_queue_, id, &blk);
  if (blk == nullptr) {
    return 1;
  }

  // 2. Wait for d2h_1st (i.e., CopyFromDev).
  CHECK_DRV(cuStreamWaitEvent(d2h_2nd_stream_, blk->evt_, 0));
  // cuMemcpyAsync((CUdeviceptr)ureq.dst, (CUdeviceptr)blk->d_ptr_, ureq.size,
  // d2h_2nd_stream_); CHECK_CUDA(cudaMemcpyAsync((void*)ureq.dst,
  // (void*)blk->d_ptr_, ureq.size, cudaMemcpyDeviceToHost, d2h_2nd_stream_));
  gmm_launch_cp_kernel(ureq.dst, blk->d_ptr_, ureq.size, d2h_2nd_stream_,
                       blk->evt_, blk->evt_);
  cuEventRecord(blk->evt_, d2h_2nd_stream_);
  // cudaStreamSynchronize(d2h_2nd_stream_);
  // cudaEventSynchronize((*blk_it)->evt_);
  *evt = blk->evt_;
  d2h_2nd_queue_.push_front(std::move(blk));
  return 0;
}

// return 0: use block to copy
//        1: direct copy
//      < 0: error
int32_t TieringCache::CopyFromHost(const UnitReq &ureq, CUevent *evt,
                                   uint64_t *id) {
  stats_["from_host"].total++;
  this->Update();

  std::unique_ptr<TieringCache::Block> blk;
  this->PopByHostPtr(rm_queue_, ureq.src, &blk);
  // Cache hit two possibilities:
  //  1. hit in rm_queue_
  //  2. hit in d2h_2nd_queue_, as the last d2h has not completed.
  if (blk) {
    *id = blk->id_;
    *evt = blk->evt_;
    // printf("-- Cache hit 11, id:%ld, h2d_2nd_queue:%zu\n", blk->id_,
    // h2d_2nd_queue_.size());
    h2d_1st_queue_.push_back(std::move(blk));
    stats_["from_host"].hit++;
    return 0;
  }
  // It is possible that the CopyToHost with the same host_ptr has not yet done,
  // which indicate the block is in d2h_2nd_queue_.
  this->PopByHostPtr(d2h_2nd_queue_, ureq.src, &blk);
  if (blk) {
    *id = blk->id_;
    *evt = blk->evt_;
    // printf("-- Cache hit 22, id:%ld, h2d_2nd_queue:%zu\n", blk->id_,
    // h2d_2nd_queue_.size());
    h2d_1st_queue_.push_back(std::move(blk));
    stats_["from_host"].hit++;
    return 0;
  }
  // printf("-- Cache Not hit, rm_queu size:%zu\n", rm_queue_.size());

  stats_["from_host"].miss++;
  // 1. Find a block.
  if (this->CanAlloc()) {
    blk = this->NewBlock();
    // 1.1 Allocate from pool if possible.
    int ret =
        pool_->alloc((CUdeviceptr *)(&blk->d_ptr_), kBlockSize_, alloc_stream_);
    if (blk->d_ptr_ == nullptr) {
      printf("Error pool alloc\n");
      abort();
    }
    size_ += kBlockSize_;
    cuEventRecord(alloc_evt_, alloc_stream_);
  } else {
    // 1.2 Reuse a block from rm_queue_.
    if (rm_queue_.empty()) {
      // printf("Warn: rm_queue empty\n");
      stats_["from_host"].direct++;
      // printf("-- %s Direct= %d, total:%zu, miss:%zu, hit:%zu\n", __func__,
      // stats_["from_host"].direct,
      //    stats_["from_host"].total, stats_["from_host"].miss,
      //    stats_["from_host"].hit);
      gmm_launch_cp_kernel(ureq.dst, ureq.src, ureq.size, ureq.stream, *evt,
                           *evt);
      return 1;
    }
    blk = std::move(rm_queue_.back());
    rm_queue_.pop_back();
  }
  // After this block finishs SwapIn and moved to rm_queue_,
  // it should not be checked for cache hit.
  blk->host_ptr = nullptr;

  // 2. Copy from host to the block.
  cuStreamWaitEvent(h2d_1st_stream_, alloc_evt_, 0);
  // cuMemcpyAsync((CUdeviceptr)ureq.src, (CUdeviceptr)blk->d_ptr_, ureq.size,
  // h2d_1st_stream_);
  CHECK_CUDA(cudaMemcpyAsync(blk->d_ptr_, ureq.src, ureq.size,
                             cudaMemcpyHostToDevice, h2d_1st_stream_));
  cuEventRecord(blk->evt_, h2d_1st_stream_);
  // cudaStreamSynchronize(h2d_1st_stream_);

  *id = blk->id_;
  *evt = blk->evt_;
  h2d_1st_queue_.push_front(std::move(blk));
  return 0;
}

int32_t TieringCache::CopyToDev(const UnitReq &ureq, CUevent *evt,
                                uint64_t id) {
  this->Update();

  std::unique_ptr<TieringCache::Block> blk;
  int ret = this->PopCompleteBlock(h2d_1st_queue_, id, &blk);
  if (ret == 2) {
    printf("h2d_1st_queue Not found block id:%ld\n", id);
    abort();
  } else if (ret == 1) {
    return 1;
  } else if (ret < 0) {
    printf("Error:%d\n", ret);
    abort();
  }

  CHECK_DRV(cuStreamWaitEvent(h2d_2nd_stream_, blk->evt_, 0));
  // cuMemcpyAsync((CUdeviceptr)ureq.dst, (CUdeviceptr)(*blk_it)->d_ptr_,
  // ureq.size, h2d_2nd_stream_);
  CHECK_CUDA(cudaMemcpyAsync(ureq.dst, blk->d_ptr_, ureq.size,
                             cudaMemcpyDeviceToDevice, h2d_2nd_stream_));
  cuEventRecord(blk->evt_, h2d_2nd_stream_);
  // cudaStreamSynchronize(h2d_2nd_stream_);

  *evt = blk->evt_;
  h2d_2nd_queue_.push_back(std::move(blk));
  return 0;
}

int32_t TieringCache::CancelPendingReq() {
  // TODO
  return 0;
}

bool TieringCache::IsBlockComplete(const TieringCache::Block &blk) {
  return cudaSuccess == cudaEventQuery(blk.evt_);
}

uint64_t TieringCache::MoveCompleteBlocks(
    std::deque<std::unique_ptr<TieringCache::Block>> &dst_q,
    std::deque<std::unique_ptr<TieringCache::Block>> &src_q) {
  uint64_t count = 0;
  while (!src_q.empty()) {
    if (!this->IsBlockComplete(*src_q.back())) {
      break;
    }
    std::unique_ptr<TieringCache::Block> blk_p = std::move(src_q.back());
    src_q.pop_back();
    dst_q.push_front(std::move(blk_p));
    count++;
  }
  return count;
}

void TieringCache::Update() {
  uint64_t ret = 0;
  // 3. Move the complete h2d 2nd blocks to removable queue.
  ret = this->MoveCompleteBlocks(rm_queue_, h2d_2nd_queue_);

  // 4. Move the complete d2h 2nd blocks to removable queue.
  ret = this->MoveCompleteBlocks(rm_queue_, d2h_2nd_queue_);
}

std::vector<UnitReq> TieringCache::DivideReq(const gmm_ipc_worker_req &req,
                                             CUstream st,
                                             const CUevent &done_evt,
                                             std::function<void(int)> cb) {
  char *const src = req.src_addr;
  char *const dst = req.tgt_addr;
  std::vector<UnitReq> ret;
  uint64_t offset = 0;
  for (; offset + kBlockSize_ < req.byte; offset += kBlockSize_) {
    UnitReq ureq = {};
    ureq.src = src + offset;
    ureq.dst = dst + offset;
    ureq.size = kBlockSize_;
    ureq.offset = offset;
    ureq.stream = st;
    ret.push_back(ureq);
  }
  // The last one.
  UnitReq ureq = {};
  ureq.src = src + offset;
  ureq.dst = dst + offset;
  ureq.size = req.src_addr + req.byte - ureq.src;
  ureq.offset = offset;
  ureq.stream = st;
  ureq.Callback = cb;
  ureq.done_evt = done_evt;
  ret.push_back(ureq);
  return ret;
}

int gmm_worker::init() {
  int ret = 0;

  ipc_dir = getenv("GMM_IPC_DIR") ? getenv("GMM_IPC_DIR") : GMM_DEFAULT_IPC_DIR;
  snprintf(gmm_socket_path, sizeof(gmm_socket_path) - 1, "%s/%s", ipc_dir,
           gmm_admin_socket);
  admin_connect = socket(AF_UNIX, SOCK_STREAM, 0);

  if (admin_connect < 0) {
    LOGGER(ERROR, "pid:%d worker:%d failed to create socket error:%s\n", pid,
           cur_dev, strerror(errno));
    ret = 1;
  }

  snprintf(log_file, 127, "/tmp/gmm-worker%d.log", cur_dev);
  gmm_log_file = fopen(log_file, "w");

  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, gmm_socket_path, sizeof(addr.sun_path) - 1);

  // connect to admin so ensure admin is ready
  while (((ret = connect(admin_connect, (struct sockaddr *)&addr,
                         sizeof(struct sockaddr_un))) != 0) &&
         (--max_retry >= 0)) {
    LOGGER(ERROR, "pid:%d worker:%d failed to connect to %s, error:%s", pid,
           cur_dev, gmm_socket_path, strerror(errno));
    sleep(1);
  }
  if (ret != 0) {
    LOGGER(ERROR, "pid:%d worker:%d failed to connect to admin", pid, cur_dev);
    ret = 2;
  }

  if ((lock_fd = open(GMM_LOCK_FILE, O_RDWR)) < 0) {
    LOGGER(ERROR, "pid:%d worker:%d failed to open %s error:%s", pid, cur_dev,
           GMM_LOCK_FILE, strerror(errno));
    ret = 3;
  }

  if ((config_fd = shm_open(GMM_CONFIG_SHM, O_RDWR, 0666)) < 0) {
    LOGGER(ERROR, "pid:%d worker:%d error open %s error:%s\n", pid, cur_dev,
           GMM_CONFIG_SHM, strerror(errno));
    ret = 4;
  }

  if ((config = (gmm_config_t *)mmap(NULL, sizeof(gmm_config_t),
                                     PROT_READ | PROT_WRITE, MAP_SHARED,
                                     config_fd, 0)) == MAP_FAILED) {
    LOGGER(ERROR, "pid:%d worker:%d mmap %s failed, error:%s", pid, cur_dev,
           GMM_CONFIG_SHM, strerror(errno));
    ret = 5;
  }

  return ret;
}

int gmm_worker::init_dm_res() {
  int ret = 0;

  dev_buf_sz = config->dev_buf_sz;
  ;
  cpu_buf_sz = config->cpu_buf_sz;
  ;
  cpu_buf = (char *)malloc(cpu_buf_sz);

  if (create_ctx) {
    CHECK_DRV(cuInit(0));
    CHECK_DRV(cuDevicePrimaryCtxRetain(&ctx, cur_dev));
    CHECK_DRV(cuCtxSetCurrent(ctx));
    LOGGER(DEBUG, "pid:%d worker:%d create ctx done", pid, cur_dev);
  } else {
    LOGGER(DEBUG, "pid:%d worker:%d create ctx skip!!!", pid, cur_dev);
  }

  /*
  LOGGER(INFO, "pid:%d worker:%d to malloc, libP:%p", pid, cur_dev, libP);
  __CF("cuMemAlloc_v2")(((CUdeviceptr*)&gpu_buf, buf_size));
  LOGGER(INFO, "pid:%d worker:%d malloc done", pid, cur_dev);
  __CF("cuMemAlloc_v2")(((CUdeviceptr*)&gpu_buf2, buf_size));
  LOGGER(INFO, "pid:%d worker:%d malloc done2", pid, cur_dev);
  */
  for (int i = 0; i < MAX_PATH_NR + 1; ++i) {
    // TODO: filter and only create stream per NVLink interconnected
    CHECK_DRV(cuStreamCreate(&dm_stream[i], CU_STREAM_NON_BLOCKING));

    for (int j = 0; j < MAX_DM_EVT_NR; ++j) {
      CHECK_DRV(cuEventCreate(&dm_evt[j], CU_EVENT_DISABLE_TIMING));
    }
    free_dm_evt_pool.fill(MAX_DM_EVT_NR);
  }

  memset(mp_evt, 0, sizeof(mp_evt));
  for (int i = 0; i < MAX_IPC_EVT_NR; ++i) {
    if (config->worker_mode !=
        GMM_MODE_DP_BIND) {  // interprocess for global or ddp mode
      CHECK_DRV(cuEventCreate(&config->gmm_evt[cur_dev].evt[i],
                              CU_EVENT_DISABLE_TIMING | CU_EVENT_INTERPROCESS));
      CHECK_DRV(cuIpcGetEventHandle(
          (CUipcEventHandle *)&config->gmm_evt[cur_dev].evt_handle[i],
          config->gmm_evt[cur_dev].evt[i]));
      LOGGER(DEBUG, "worker:%d IPC evt:%p", cur_dev,
             config->gmm_evt[cur_dev].evt[i]);
    } else {
      CHECK_DRV(cuEventCreate(&config->gmm_evt[cur_dev].evt[i],
                              CU_EVENT_DISABLE_TIMING));
      LOGGER(DEBUG, "worker:%d same-process evt:%p", cur_dev,
             config->gmm_evt[cur_dev].evt[i]);
    }
  }
  config->gmm_evt[cur_dev].creator_pid = pid;

  CUmemAccessDesc accessDesc;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  accessDesc.location.id = cur_dev;
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

  CHECK_DRV(cuMemGetInfo(&dev_free_sz, &dev_tot_sz));
  if (init_dev_memPool_sz > dev_free_sz) init_dev_memPool_sz = dev_free_sz;

  if (max_dev_memPool_sz > dev_tot_sz) max_dev_memPool_sz = dev_tot_sz;

  // Now assume only GPU 0 needs SwapOut/In.
  // TODO: unified cache management.
  // if (cur_dev != 0) {
  uint64_t max_size = dev_free_sz * 1 / 10;
  uint64_t block_size = 1ULL << 23;
  const char *env_p = std::getenv("GLAKE_CACHE_MAX_SIZE");
  if (env_p) {
    // Unit: MB.
    max_size = std::stoi(env_p) * (1ULL << 20);
    // printf("Get env GLAKE_CACHE_MAX_SIZE:%ld MB\n", max_size>>20);
    env_p = nullptr;
  }
  env_p = std::getenv("GLAKE_CACHE_BLOCK_SIZE");
  if (env_p) {
    // Unit: MB.
    block_size = std::stoi(env_p) * (1ULL << 20);
    // printf("Get env GLAKE_CACHE_BLOCK_SIZE:%ld MB\n", block_size>>20);
  }
  tcache_ = std::make_unique<TieringCache>(block_size, max_size);

  // Now this pool is only for GLake cache. Maybe we can use the same pool for
  // client cudaMalloc in the future.
  dev_memPool = new gmm_memPool(cur_dev, max_size, max_size);
  tcache_->SetMemPool(dev_memPool);
  //}
  return ret;
}

int gmm_worker::register_and_serve() {
  int ret = 0;
  gmm_ipc_admin_req req;
  req.data.newDev_req.pid = pid;
  req.data.newDev_req.dev_id = cur_dev;
  // CHECK_DRV(cuDeviceGetPCIBusId(req.data.newDev_req.dev_bus, 20, cur_dev));
  req.op = GMM_OP_NEW_WORKER;

  // reigster to admin
  if (gmm_send(admin_connect, (void *)&req, sizeof(req)) > 0 &&
      gmm_recv(admin_connect, (void *)&ret, sizeof(ret)) > 0 && ret == 0) {
  } else {
    LOGGER(ERROR, "pid:%d worker:%d failed to register to GMM admin", pid,
           cur_dev);
    return 1;
  }

  // start service
  if ((socket_fd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
    LOGGER(ERROR, "pid:%d worker:%d failed to create socket error:%s", pid,
           cur_dev, strerror(errno));
    return 2;
  }

  snprintf(gmm_socket_path, sizeof(gmm_socket_path) - 1,
           "%s/gmm_worker_%d_%d.sock", ipc_dir, pid, cur_dev);
  strncpy(addr.sun_path, gmm_socket_path, sizeof(addr.sun_path));
  unlink(gmm_socket_path);
  if (bind(socket_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    LOGGER(ERROR, "pid:%d worker:%d failed to bind error:%s", pid, cur_dev,
           strerror(errno));
    return 3;
  }

  if (listen(socket_fd, MAX_CLIENT_NUM) < 0) {
    LOGGER(ERROR, "pid:%d worker:%d failed to listen error:%s", pid, cur_dev,
           strerror(errno));
    return 4;
  }

  struct flock f_lock;
  f_lock.l_type = F_WRLCK;
  f_lock.l_whence = SEEK_SET;
  f_lock.l_start = 0;
  f_lock.l_len = 0;

  fcntl(lock_fd, F_SETLKW, &f_lock);
  config->ready_cnt++;
  f_lock.l_type = F_UNLCK;
  fcntl(lock_fd, F_SETLKW, &f_lock);
  close(lock_fd);

  config->worker_creator[cur_dev] = pid;

  LOGGER(INFO,
         "pid:%d tid:%d worker:%d started, GPU mem freeMB:%ld totMB:%ld log:%s "
         "path:%s ctx:%d",
         pid, gettid(), cur_dev, dev_free_sz >> 20, dev_tot_sz >> 20, log_file,
         gmm_socket_path, create_ctx);

  // gmm_set_log_level();

  return ret;
}

// handler for SG
int gmm_worker::dm_sg_handler(gmm_ipc_worker_req &req, CUevent &pre_evt,
                              CUevent &done_evt) {
  int ret = 0;

  LOGGER(DEBUG,
         "worker:%d pid:%d req.pid:%d src:%p tgt:%p async:%d config.sync:%d "
         "pre_evt:%p done_evt:%p",
         cur_dev, pid, req.pid, req.src_addr, req.tgt_addr, req.async,
         config->sync, pre_evt, done_evt);

  switch (req.gmm_op) {
    case GMM_OP_SCATTER: {
      CUdeviceptr tgt_addr = {};
      if (req.dm_type == GMM_DM_KERNEL) {
        ret = dev_memPool->alloc(&tgt_addr, req.byte, get_stream(req.src_dev));
        if (ret == 0) {
          insert_sg_info(req.gid, req.slot_id, req.tgt_addr, req.byte,
                         req.split_offset_src, req.base_offset_src, tgt_addr);
#if 0
          // Scatter gather kernel
          printf("[W] Scatter K1 tgt:%p src:%p\n", (void *)tgt_addr, req.src_addr);
          gmm_launch_cp_kernel((char *)tgt_addr, req.src_addr, req.byte, get_stream(req.src_dev), pre_evt, done_evt, config->sync);
          printf("[W] Scatter K2 tgt:%p src:%p\n", req.tgt_addr, (void *)tgt_addr);
          gmm_launch_cp_kernel((char *)req.tgt_addr, (char *)tgt_addr, req.byte, get_stream(req.src_dev), pre_evt, done_evt, config->sync);
#endif
#if 0
          // Scatter gather DMA
          CHECK_DRV(cuStreamWaitEvent(get_stream(req.src_dev), pre_evt, CU_EVENT_WAIT_DEFAULT));
          CHECK_CUDA(cudaMemcpyAsync((char *)tgt_addr, req.src_addr, req.byte, cudaMemcpyDefault, get_stream(req.src_dev)));
          CHECK_CUDA(cudaMemcpyAsync(req.tgt_addr, (char *)tgt_addr, req.byte, cudaMemcpyDefault, get_stream(req.src_dev)));
          CHECK_DRV(cuEventRecord(done_evt, get_stream(req.src_dev)));
#endif
#if 1
          // Multi-path kernel
          // auto t0  = std::chrono::steady_clock::now();
          gmm_launch_cp_kernel((char *)req.tgt_addr, (char *)req.src_addr,
                               req.byte, get_stream(req.src_dev), pre_evt,
                               done_evt, config->sync);
          // cuEventSynchronize(done_evt);
          // auto t1  = std::chrono::steady_clock::now();
          // auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t1
          // -t0).count();
#endif
        }
      }
      // TODO else DMA
      break;
    }

    case GMM_OP_GATHER: {
      // re-use pre/post evt?
      if (req.dm_type == GMM_DM_KERNEL) {
        CUdeviceptr cur_addr = {};
        if (find_sg_info(req.slot_id, req.src_addr, cur_addr) == 0) {
          if (cur_addr == 0) {
            printf("-- find_sg_info cur_addr nullptr\n");
            abort();
          }
#if 0
          // Scatter gather kernel
          printf("[W] Gather K1 tgt:%p src:%p\n", (void *)cur_addr, req.src_addr);
          gmm_launch_cp_kernel((char *)cur_addr, req.src_addr, req.byte, get_stream(req.tgt_dev), pre_evt, done_evt, config->sync);
          printf("[W] Gather K2 tgt:%p src:%p\n", req.tgt_addr, (void *)cur_addr);
          gmm_launch_cp_kernel(req.tgt_addr, (char *)cur_addr, req.byte, get_stream(req.tgt_dev), pre_evt, done_evt, config->sync);
#endif
#if 0
          // Scatter gather DMA
          CHECK_DRV(cuStreamWaitEvent(get_stream(req.tgt_dev), pre_evt, CU_EVENT_WAIT_DEFAULT));
          CHECK_CUDA(cudaMemcpyAsync((char *)cur_addr, req.src_addr, req.byte, cudaMemcpyDefault, get_stream(req.tgt_dev)));
          CHECK_CUDA(cudaMemcpyAsync(req.tgt_addr, (char *)cur_addr, req.byte, cudaMemcpyDefault, get_stream(req.tgt_dev)));
          CHECK_DRV(cuEventRecord(done_evt, get_stream(req.tgt_dev)));
#endif
#if 1
          // Multi-path kernel
          // auto t0  = std::chrono::steady_clock::now();
          gmm_launch_cp_kernel(req.tgt_addr, (char *)req.src_addr, req.byte,
                               get_stream(req.tgt_dev), pre_evt, done_evt,
                               config->sync);
          // auto t1  = std::chrono::steady_clock::now();
          // cuEventSynchronize(done_evt);
          // auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t1
          // -t0).count();
#endif
        }

        if (config->sync) {
          delete_sg_info(req.slot_id, req.src_addr);
        }
      }
      // TODO else
      break;
    }
    default: {
      break;
    }
  }

  return ret;
}

// Tiering Cache handler
// This handler only divides the requests, not really dispatch them to GPU.
int gmm_worker::dm_cache_handler(gmm_ipc_worker_req &req, CUevent &pre_evt,
                                 CUevent &done_evt, int conn_fd) {
  int ret = 0;
  LOGGER(DEBUG,
         "worker:%d pid:%d req.pid:%d src:%p tgt:%p async:%d config.sync:%d "
         "pre_evt:%p done_evt:%p",
         cur_dev, pid, req.pid, req.src_addr, req.tgt_addr, req.async,
         config->sync, pre_evt, done_evt);

  switch (req.gmm_op) {
    case GMM_OP_SCATTER: {
      CHECK_DRV(cuStreamWaitEvent(get_stream(req.src_dev), pre_evt,
                                  CU_EVENT_WAIT_DEFAULT));

      // 0. Cancel pending req.
      tcache_->CancelPendingReq();

      // 1. divide req by unit, insert them to 1st ureq queue.
      std::function<void(int)> final_callback = [conn_fd](int ret) -> void {
        gmm_send(conn_fd, &ret, sizeof(ret));
      };
      std::vector<UnitReq> ureqs = tcache_->DivideReq(
          req, get_stream(req.src_dev), done_evt, final_callback);
      // printf("[W%d] divide %zu blocks\n", cur_dev, ureqs.size());
      for (auto ureq_it = ureqs.begin(); ureq_it != ureqs.end(); ureq_it++) {
        scatter_1st_reqs_.push_front(*ureq_it);
      }
      break;
    }
    case GMM_OP_GATHER: {
      CHECK_DRV(cuStreamWaitEvent(get_stream(req.tgt_dev), pre_evt,
                                  CU_EVENT_WAIT_DEFAULT));

      // 1. divide req, insert to 1st req queue.
      std::function<void(int)> final_callback = [conn_fd](int ret) -> void {
        gmm_send(conn_fd, &ret, sizeof(ret));
      };

      std::vector<UnitReq> ureqs = tcache_->DivideReq(
          req, get_stream(req.tgt_dev), done_evt, final_callback);
      // printf("-- dividie:%zu\n", ureqs.size());
      for (auto ureq_it = ureqs.begin(); ureq_it != ureqs.end(); ureq_it++) {
        gather_1st_reqs_.push_front(*ureq_it);
      }
      break;
    }
    default: {
      break;
    }
  }

  return ret;
}

// handler for DM req
int gmm_worker::dm_handler(gmm_ipc_worker_req &req, CUevent &pre_evt,
                           CUevent &done_evt) {
  int ret = 0;

  char *src_addr = req.src_addr;
  char *tgt_addr = req.tgt_addr;

  LOGGER(DEBUG,
         "worker:%d pid:%d req.pid:%d src:%p tgt:%p async:%d config.sync:%d "
         "pre_evt:%p done_evt:%p",
         cur_dev, pid, req.pid, req.src_addr, req.tgt_addr, req.async,
         config->sync, pre_evt, done_evt);

  // setup src and tgt addr if IPC
  if (req.cross_process) {
    gmm_shmInfo_t *shm_host = (gmm_shmInfo_t *)req.shmInfo_addr_src;
    src_addr =
        ((char *)shm_host->addr) + req.base_offset_src + req.split_offset_tgt;

    gmm_shmInfo_t *shm_dev = (gmm_shmInfo_t *)req.shmInfo_addr_tgt;
    tgt_addr =
        ((char *)shm_dev->addr) + req.base_offset_tgt + req.split_offset_tgt;
  }

  // sanity test on src and tgt
  if (0 && cur_dev != 0) {
    int sz = (test_sz > req.byte) ? req.byte : test_sz;
    char *test = (char *)malloc(sz);

    if (req.gmm_op == GMM_OP_H2D) {
      memcpy(test, src_addr, sz);
      LOGGER(DEBUG, "worker:%d pid:%d test read on host addr:%p done", cur_dev,
             pid, src_addr);
      gmm_launch_cp_kernel(test, tgt_addr, sz, get_stream(req.tgt_dev), pre_evt,
                           done_evt, true);
      LOGGER(DEBUG, "worker:%d pid:%d test read on dev addr:%p done", cur_dev,
             pid, tgt_addr);
    } else if (req.gmm_op == GMM_OP_D2H) {
      memcpy(test, tgt_addr, sz);
      LOGGER(DEBUG, "worker:%d pid:%d test read on host addr:%p done", cur_dev,
             pid, tgt_addr);
      gmm_launch_cp_kernel(test, src_addr, sz, get_stream(req.src_dev), pre_evt,
                           done_evt, true);
      LOGGER(DEBUG, "worker:%d pid:%d test read on dev addr:%p done", cur_dev,
             pid, src_addr);
    } else if (req.gmm_op == GMM_OP_D2D) {
      gmm_launch_cp_kernel(test, tgt_addr, sz, get_stream(req.tgt_dev), pre_evt,
                           done_evt, true);
      LOGGER(DEBUG, "worker:%d pid:%d test read on dev addr:%p done", cur_dev,
             pid, tgt_addr);
    }

    if (test) free(test);
  }

  switch (req.gmm_op) {
    case GMM_OP_H2D: {
      if (req.dm_type == GMM_DM_KERNEL) {
        gmm_launch_cp_kernel(tgt_addr, src_addr, req.byte, get_inStream(),
                             pre_evt, done_evt, config->sync);
      } else {
        if (cur_dev == req.src_dev) {
          gmm_launch_cp_DMA(req.tgt_addr, req.src_addr, gpu_buf, req.byte,
                            dev_buf_sz, get_inStream(), pre_evt, done_evt,
                            false);
        } else {
          // gmm_pipeline_DMA(req.tgt_addr, req.src_addr, req.byte, gpu_buf,
          // gpu_buf2, dev_buf_sz, get_inStream(), get_stream(req.src_dev),
          // h2d_e1, h2d_e2, pre_evt, done_evt, false);
        }
      }
      break;
    }

    case GMM_OP_D2H: {
      if (req.dm_type == GMM_DM_KERNEL) {
        gmm_launch_cp_kernel(tgt_addr, src_addr, req.byte, get_outStream(),
                             pre_evt, done_evt, config->sync);
      }
      // else gmm_DMA_direct(tgt_addr, src_addr, req.byte, get_outStream(),
      // pre_evt, done_evt, false);
      break;
    }

    case GMM_OP_D2D: {
      if (req.dm_type == GMM_DM_KERNEL) {
        gmm_launch_cp_kernel(tgt_addr, src_addr, req.byte,
                             get_stream(req.tgt_dev), pre_evt, done_evt,
                             config->sync);
      }
      // else gmm_DMA_direct(tgt_addr, src_addr, req.byte, get_outStream(),
      // pre_evt, done_evt, false);
      break;
    }

    default: {
      break;
    }
  }

  return ret;
}

int gmm_worker::register_cuda_shm_handler(gmm_ipc_worker_req &req,
                                          gmm_shmInfo_t *&shm_out,
                                          shared_fd fd) {
  int ret = 0;

  CUdeviceptr dptr;
  CHECK_DRV(cuMemAddressReserve(&dptr, req.byte, 0ULL, 0U, 0));
  CUmemGenericAllocationHandle handle;
  CHECK_DRV(
      cuMemImportFromShareableHandle(&handle, (void *)(uintptr_t)fd,
                                     CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  CHECK_DRV(cuMemMap((CUdeviceptr)dptr, req.byte, 0ULL, handle, 0ULL));

  accessDesc.location.id = cur_dev;
  CHECK_DRV(cuMemSetAccess((CUdeviceptr)dptr, req.byte, &accessDesc, 1));
  gmm_shmInfo_t *shm = new gmm_shmInfo_t(GMM_IPC_MEM_NV_DEV_SHARE, req.src_dev,
                                         (void *)dptr, handle, req.byte, fd);
  LOGGER(INFO, "worker:%d pid:%d req.pid:%d register dev shm send rsp", cur_dev,
         pid, req.pid);
  shm_out = shm;

  return ret;
}

int gmm_worker::register_cpu_shm_handler(gmm_ipc_worker_req &req,
                                         gmm_shmInfo_t *&shm_out) {
  int ret = 0;

  gmm_shmInfo_t *shm = new gmm_shmInfo_t(GMM_IPC_MEM_HOST_PIN_SHARE, CPU_DEV,
                                         req.pid, req.shm_idx, req.byte);
  char shm_name[MAX_SHM_PATH_LEN];
  snprintf(shm_name, sizeof(shm_name) - 1, "%s_%d_%d", GMM_HOST_SHM_PREFIX,
           shm->get_pid(), shm->get_idx());
  ret = gmm_shmOpen(shm_name, req.byte, shm);
  CHECK_DRV(cuMemHostRegister(
      shm->get_addr(), req.byte,
      CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP));

  LOGGER(INFO, "worker:%d pid:%d req.pid:%d register dev shm send rsp", cur_dev,
         pid, req.pid);
  shm_out = shm;

  return ret;
}

static void *gmm_worker_proc(void *args) {
  worker_args *arg = (worker_args *)args;
  bool create_ctx = arg->create_ctx;
  int cur_dev = arg->cur_dev;
  int launcher_dev = arg->launcher_dev;
  pid_t ppid = arg->launcher_pid;
  pid_t pid = getpid();
  int ret = 0;

  thread_local auto start_t = std::chrono::steady_clock::now();
  gmm_worker worker(arg);
  ret = worker.init();
  ret = worker.init_dm_res();
  ret = worker.register_and_serve();

  int socket_fd = worker.get_socket();
  fd_set active_fd_set, read_fd_set;
  FD_ZERO(&active_fd_set);
  FD_SET(socket_fd, &active_fd_set);
  struct timeval timeout;
  timeout.tv_sec = 0;
  timeout.tv_usec = 2;

  std::mutex mtx;
  gmm_config_t *config = worker.get_config();
  size_t dev_buf_sz = worker.get_gpuBuf_size();
  CUstream in_stream = worker.get_inStream();
  CUstream out_stream = worker.get_outStream();

  arg->ready = GMM_STATE_WORKER_READY;
  // printf("gmm worker:%d pid:%d started log:%d\n", cur_dev, pid,
  // gmm_log_level);
  while (true) {
    read_fd_set = active_fd_set;

    int ret = select(FD_SETSIZE, &read_fd_set, NULL, NULL, &timeout);
    if (ret >= 0) {
      for (int i = 0; i < FD_SETSIZE; i++) {
        if (FD_ISSET(i, &read_fd_set)) {
          if (i == socket_fd) {  // new connect
            int new_socket = accept(socket_fd, NULL, NULL);
            if (new_socket < 0) {
              continue;
            }
            // LOGGER(INFO, "worker:%d new connect id:%d", cur_dev, new_socket);
            FD_SET(new_socket, &active_fd_set);
          } else {
            // active then handling the req
            gmm_ipc_worker_req req = {};
            shared_fd fd;
            mtx.lock();

            // auto t0  = std::chrono::steady_clock::now();
            // int ret = recv(i, &req, sizeof(gmm_ipc_worker_req), 0);
            int ret =
                gmm_recv(i, &req, sizeof(req), GMM_OP_REGISTER_SHM_DEV, &fd);
            // auto t1  = std::chrono::steady_clock::now();
            // auto duration =
            // std::chrono::duration_cast<std::chrono::microseconds>(t1
            // -t0).count();
            // std::cout << "-- recv dur:" << duration << " us\n";
            if (ret > 0) {  // data comes

              CUevent pre_evt = nullptr, done_evt = nullptr;
              worker.prepare_evt(req, pre_evt, done_evt);

              switch (req.gmm_op) {
                case GMM_OP_H2D: {
                  // printf("[W:%d] op H2D size:%zu MB\n", cur_dev,
                  // req.byte>>20UL);
                  ret = worker.dm_handler(req, pre_evt, done_evt);
                  gmm_send(i, &ret, sizeof(ret));
                  break;
                }

                case GMM_OP_D2H: {
                  // printf("[W:%d] op D2H\n", cur_dev);
                  ret = worker.dm_handler(req, pre_evt, done_evt);
                  gmm_send(i, &ret, sizeof(ret));
                  break;
                }

                case GMM_OP_D2D: {
                  printf("[W:%d] op D2D\n", cur_dev);
                  ret = worker.dm_handler(req, pre_evt, done_evt);
                  gmm_send(i, &ret, sizeof(ret));
                  break;
                }

                case GMM_OP_REGISTER_SHM_DEV: {
                  gmm_shmInfo_t *shm = nullptr;
                  ret = worker.register_cuda_shm_handler(req, shm, fd);

                  gmm_ipc_worker_rsp rsp(ret, (uint64_t)shm);
                  gmm_send(i, &rsp, sizeof(rsp));
                  break;
                }

                case GMM_OP_REGISTER_SHM_HOST: {
                  gmm_shmInfo_t *shm = nullptr;
                  ret = worker.register_cpu_shm_handler(req, shm);

                  gmm_ipc_worker_rsp rsp(ret, (uint64_t)shm);
                  gmm_send(i, &rsp, sizeof(rsp));
                  break;
                }

                case GMM_OP_DEREGISTER_SHM_HOST: {
                  gmm_shmInfo_t *shm = (gmm_shmInfo_t *)req.shmInfo_addr_src;
                  if (shm) {
                    cuMemHostUnregister(shm->get_addr());
                    gmm_shmClose(shm);
                    // shmHost_table.del_shmEntry(req.src_addr);
                  }
                  break;
                }

                case GMM_OP_DEREGISTER_SHM_DEV: {
                  gmm_shmInfo_t *shm = (gmm_shmInfo_t *)req.shmInfo_addr_src;
                  if (shm) {
                    CHECK_DRV(cuMemUnmap((CUdeviceptr)shm->get_addr(),
                                         shm->get_size()));
                    CHECK_DRV(cuMemAddressFree((CUdeviceptr)shm->get_addr(),
                                               shm->get_size()));
                    CHECK_DRV(cuMemRelease(shm->get_handle()));
                    close(shm->get_shmFd());
                  }
                  // no rsp
                  break;
                }

                case GMM_OP_SCATTER: {
                  // printf("-- [Worker %d] Req src:[%p %p], dst:[%p %p],
                  // size:%zu KB\n", cur_dev,
                  //    req.src_addr, req.src_addr+req.byte, req.tgt_addr,
                  //    req.tgt_addr+req.byte, req.byte>>10UL);

                  // start_t = std::chrono::steady_clock::now();
                  if (worker.tcache_) {
                    // The Local GPU does not need cache.
                    // auto t0  = std::chrono::steady_clock::now();
                    worker.dm_cache_handler(req, pre_evt, done_evt, i);
                    // auto t1  = std::chrono::steady_clock::now();
                    // cuEventSynchronize(done_evt);
                    // auto dur =
                    // std::chrono::duration_cast<std::chrono::microseconds>(t1
                    // -t0).count(); printf("Dev:%d SwapOut:%zu MB, us:%ld\n",
                    // cur_dev, req.byte>>20UL, dur);
                  } else {
                    ret = worker.dm_sg_handler(req, pre_evt, done_evt);
                    gmm_send(i, &ret, sizeof(ret));
                  }
                  break;
                }

                case GMM_OP_GATHER: {
                  if (worker.tcache_) {
                    // The Local GPU does not need cache.
                    // auto t0  = std::chrono::steady_clock::now();
                    worker.dm_cache_handler(req, pre_evt, done_evt, i);
                    // auto t1  = std::chrono::steady_clock::now();
                    // cuEventSynchronize(done_evt);
                    // auto dur =
                    // std::chrono::duration_cast<std::chrono::microseconds>(t1
                    // -t0).count(); printf("Dev:%d SwapIn:%zu MB, us:%ld\n",
                    // cur_dev, req.byte>>20UL, dur);
                  } else {
                    ret = worker.dm_sg_handler(req, pre_evt, done_evt);
                    gmm_send(i, &ret, sizeof(ret));
                  }
                  break;
                }

                case GMM_OP_STOP: {
                  LOGGER(ERROR, "To support");
                  break;
                }

                default: {
                  LOGGER(ERROR, "worker:%d invalid op:%d", cur_dev, req.gmm_op);
                  break;
                }
              }
            } else if (ret == 0 && errno != EINTR) {
              LOGGER(INFO, "admin: client:%d read 0", i);
              // gmm_mem_cleanup(dev, i, borrow_sz);
              close(i);
              FD_CLR(i, &active_fd_set);
            } else {  // exception
              LOGGER(INFO, "admin: client:%d read ret -1", i);
              // gmm_mem_cleanup(dev, i, borrow_sz);
              close(i);
              FD_CLR(i, &active_fd_set);
            }
            mtx.unlock();
          }
        }  // if FD_ISSET
      }    // For FD_SETSIZE
    }      // select ret>0

    // GPU 0 does not have cache.
    if (worker.tcache_ == nullptr) {
      continue;
    }
    /*
     *  Tiering cache operations.
     */

    CUevent copy_evt = {};
    // Gather 1st.
    if (IsPcieRxNotBusy() && !worker.gather_1st_reqs_.empty()) {
      UnitReq ureq = worker.gather_1st_reqs_.back();
      ret = worker.tcache_->CopyFromHost(ureq, &copy_evt, &ureq.id);
      if (ret == 0) {
        worker.gather_1st_reqs_.pop_back();
        worker.gather_2nd_reqs_.push_front(ureq);
      } else if (ret == 1) {
        worker.gather_1st_reqs_.pop_back();
        if (ureq.IsFinal()) {
          worker.tcache_->PrintStats(cur_dev, "from_host");
          worker.tcache_->ResetStats("from_host");
          cuStreamWaitEvent(ureq.stream, copy_evt, 0);
          cuEventRecord(ureq.done_evt, ureq.stream);
          ureq.Callback(ret);
        }
      } else {
        printf("Error: CopyFromHost %d\n", ret);
      }
    }
    // Gather 2nd.
    if (IsNvlinkTxNotBusy() && !worker.gather_2nd_reqs_.empty()) {
      UnitReq ureq = worker.gather_2nd_reqs_.back();
      ret = worker.tcache_->CopyToDev(ureq, &copy_evt, ureq.id);
      // Success
      if (ret == 0) {
        worker.gather_2nd_reqs_.pop_back();
        if (ureq.IsFinal()) {
          worker.tcache_->PrintStats(cur_dev, "from_host");
          worker.tcache_->ResetStats("from_host");
          cuStreamWaitEvent(ureq.stream, copy_evt, 0);
          cuEventRecord(ureq.done_evt, ureq.stream);
          ureq.Callback(ret);
        }
      } else if (ret < 0) {
        printf("Error: CopyToDev %d\n", ret);
        abort();
      } else if (ret == 1) {
        // Do nothing. As the 1st stage blocks has not completed.
      } else {
        printf("Not supported ret:%d\n", ret);
        abort();
      }
    }

    // Scatter 1st.
    if (IsNvlinkRxNotBusy() && !worker.scatter_1st_reqs_.empty()) {
      UnitReq ureq = worker.scatter_1st_reqs_.back();
      ret = worker.tcache_->CopyFromDev(ureq, &copy_evt, &ureq.id);
      if (ret == 0) {
        worker.scatter_1st_reqs_.pop_back();
        worker.scatter_2nd_reqs_.push_front(ureq);
      } else if (ret == 1) {
#if 0  // Debug
          if (ureq.id == 1) {
            auto t = std::chrono::steady_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t - start_t).count();
            auto g_dur = std::chrono::duration_cast<std::chrono::microseconds>(t - g_req_start_t).count();
            std::cout << "[W" << cur_dev << "] 1st CopyFromDev:" << dur << " us, g_dur: " << g_dur << " us\n";
          }
#endif
        worker.scatter_1st_reqs_.pop_back();

#if 0  // Debug
            auto t = std::chrono::steady_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t - start_t).count();
            auto g_dur = std::chrono::duration_cast<std::chrono::microseconds>(t - g_req_start_t).count();
            std::cout << "[W" << cur_dev << "] " << ureq.id << "th CopyFromDev:" << dur << " us, g_dur: " <<
              g_dur << " us\n";

            cuEventSynchronize(copy_evt);
            t = std::chrono::steady_clock::now();
            dur = std::chrono::duration_cast<std::chrono::microseconds>(t - start_t).count();
            g_dur = std::chrono::duration_cast<std::chrono::microseconds>(t - g_req_start_t).count();
            std::cout << "[W" << cur_dev << "] " << ureq.id << "th Sync:" << dur << " us, g_dur: " <<
              g_dur << " us\n";
#endif
      } else {
        printf("-- Error: CopyFromDev ret:%d\n", ret);
        abort();
      }
      if (ureq.IsFinal()) {
        cuStreamWaitEvent(ureq.stream, copy_evt, 0);
        cuEventRecord(ureq.done_evt, ureq.stream);
        ureq.Callback(ret);
      }
    }
#if 1
    // Scatter 2nd.
    // Simutaneously running D2D and D2H makes D2D slow, which affects client
    // sync performance.
    //   To alleviate this problem, D2H is delayed for some round.
    // TODO: Determine the delay time. It cannot be too short due to
    // performance. It also
    //  cannot be too long, as it increases cache full possibilities.
    const uint32_t max_wait_n = 100;
    thread_local uint32_t wait_n = 0;
    if (IsPcieTxNotBusy() && !worker.scatter_2nd_reqs_.empty() &&
        wait_n++ > max_wait_n) {
      UnitReq ureq = worker.scatter_2nd_reqs_.back();
      int ret = worker.tcache_->CopyToHost(ureq, &copy_evt, ureq.id);
      if (ret == 0) {
        // Success.
        worker.scatter_2nd_reqs_.pop_back();
      } else if (ret > 0) {
        // Skip this round.
      } else if (ret < 0) {
        printf("Error: CopyToHost %d\n", ret);
        abort();
      }
    }
#endif

    if (IsLocalGpuMemTight()) {
      int64_t cur_free = GetGpuFreeSize();
      worker.tcache_->Shrink(cur_free / 2);
    }
  }  // while

worker_error:
  arg->ready = GMM_STATE_WORKER_ERROR;
  return nullptr;
}

// start workers if doesn't exist
gmm_state gmm_launch_worker_thr(int cur_dev, int worker_dev, bool create_ctx) {
  worker_args arg = {.launcher_pid = getpid(),
                     .launcher_dev = cur_dev,
                     .cur_dev = worker_dev,
                     .ready = GMM_STATE_INIT,
                     .create_ctx = create_ctx};
  pthread_t thr;
  int ret = pthread_create(&thr, NULL, gmm_worker_proc, &arg);
  if (ret != 0) {
    LOGGER(INFO, "failed to create worker thread error:%s", strerror(errno));
  }
  std::string name{"Worker_"};
  name += std::to_string(worker_dev);
  pthread_setname_np(thr, name.c_str());

  while (arg.ready == GMM_STATE_INIT) {
    sched_yield();
  }

  return arg.ready;
}
