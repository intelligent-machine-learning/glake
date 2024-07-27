#pragma once
#include <sys/mman.h>
#include <sys/wait.h>

#include <deque>
#include <memory>
#include <unordered_map>

#include "gmm_common.h"
#include "gmm_cuda_common.h"
#include "gmm_cuda_mempool.h"
#include "gmm_host_shm.h"
#include "gmm_queue.h"
#include "gmm_vstore.h"

// worker thread per GPU
// TODO: register to admin?, when admin exit, drain worker before exit
struct worker_args {
  pid_t launcher_pid;  // launcher pid
  int launcher_dev;    // launcher process's cur_dev
  int cur_dev;         // worker dev
  gmm_state ready;
  size_t init_dev_memPool_sz;
  size_t max_dev_memPool_sz;
  bool create_ctx;
};

class UnitReq {
 public:
  UnitReq() = default;
  ~UnitReq() = default;

  bool IsFinal() const { return Callback != nullptr; }
  std::function<void(int)> Callback;

  enum CopyType { D2H, H2D };
  char *src;
  char *dst;
  uint64_t size;
  // Relative to the complete Request.
  uint64_t offset;
  CUstream stream;
  CopyType type;
  // To identify the 1st and the 2nd ureqs which belong to the same req.
  uint64_t id;
  CUevent done_evt;
};

class TieringCache {
 public:
  TieringCache(uint64_t blk_size, uint64_t max_size)
      : kBlockSize_(blk_size),
        max_size_(max_size),
        copy_from_host_n_(0),
        miss_n_(0),
        hit_n_(0),
        id_(1) {
    cuEventCreate(&alloc_evt_, CU_EVENT_DISABLE_TIMING);
    Stat stat = {};
    stats_["from_host"] = stat;
    stats_["from_dev"] = stat;
    stats_["to_host"] = stat;
    stats_["to_dev"] = stat;
  }
  ~TieringCache() { cuEventDestroy(alloc_evt_); }

  void SetMemPool(gmm_memPool *p) { pool_ = p; }
  uint64_t Size() const { return size_; }

  int32_t Shrink(uint64_t size);
  // id is used to match CopyToHost.
  int32_t CopyFromDev(const UnitReq &ureq, CUevent *done_evt, uint64_t *id);
  int32_t CopyToHost(const UnitReq &ureq, CUevent *done_evt, const uint64_t id);
  // If cache hit, reserve the hit block and return without PCIe copy.
  int32_t CopyFromHost(const UnitReq &ureq, CUevent *done_evt, uint64_t *id);
  int32_t CopyToDev(const UnitReq &ureq, CUevent *done_evt, const uint64_t id);
  int32_t CancelPendingReq();

  std::vector<UnitReq> DivideReq(const gmm_ipc_worker_req &req, CUstream st,
                                 const CUevent &done_evt,
                                 std::function<void(int)> cb);
  bool IsCacheHit(char *host_ptr, uint64_t *blk_id) const;

  void PrintStats(int gpu, const std::string &k) {
    printf("Worker %d Stat %s: total=%ld hit=%ld miss=%ld direct=%ld\n", gpu,
           k.c_str(), stats_[k].total, stats_[k].hit, stats_[k].miss,
           stats_[k].direct);
  }
  void ResetStats(const std::string &k) { stats_[k] = Stat{}; }

 private:
  class Block {
   public:
    Block(uint64_t id, uint64_t size)
        : d_ptr_(nullptr), size_(size), id_(id), host_ptr(nullptr) {
      cudaEventCreateWithFlags(&evt_, cudaEventDisableTiming);
    }
    ~Block() { cudaEventDestroy(evt_); }

    // The GPU buffer belonging to this block.
    char *d_ptr_;
    uint64_t size_;
    cudaEvent_t evt_;
    uint64_t id_;
    // host_ptr is used as the key to match SwapOut and SwapIn,
    // whcih is used for cache hit. Maybe we need to add slot_id
    // in order to differentiate address across processes.
    char *host_ptr;
  };

  struct Stat {
    uint64_t total;
    uint64_t miss;
    uint64_t hit;
    // direct copy
    uint64_t direct;
  };
  // Scan all 2nd_queue_. Move the complete blocks to rm_queue_.
  void Update();
  std::unique_ptr<Block> NewBlock() {
    return std::make_unique<Block>(this->GenId(), kBlockSize_);
  }
  uint64_t GenId() { return id_++; }
  bool IsBlockComplete(const TieringCache::Block &blk);
  int32_t PopCompleteBlock(std::deque<std::unique_ptr<Block>> &queue,
                           uint64_t blk_id, std::unique_ptr<Block> *blk);
  void PopByHostPtr(std::deque<std::unique_ptr<TieringCache::Block>> &queue,
                    const char *const host_p,
                    std::unique_ptr<TieringCache::Block> *blk);
  uint64_t MoveCompleteBlocks(std::deque<std::unique_ptr<Block>> &dst_q,
                              std::deque<std::unique_ptr<Block>> &src_q);
  bool CanAlloc() const { return size_ < max_size_; }

  const uint64_t kBlockSize_;
  gmm_memPool *pool_;
  uint64_t size_;
  uint64_t max_size_;
  // FIFO deque: push_front, pop_back.
  std::deque<std::unique_ptr<Block>> rm_queue_;
  std::deque<std::unique_ptr<Block>> h2d_1st_queue_;
  std::deque<std::unique_ptr<Block>> h2d_2nd_queue_;
  std::deque<std::unique_ptr<Block>> d2h_1st_queue_;
  std::deque<std::unique_ptr<Block>> d2h_2nd_queue_;
  CUstream h2d_1st_stream_;
  CUstream h2d_2nd_stream_;
  CUstream d2h_1st_stream_;
  CUstream d2h_2nd_stream_;
  CUstream alloc_stream_;
  CUevent alloc_evt_;
  uint64_t copy_from_host_n_;
  uint64_t miss_n_;
  uint64_t hit_n_;
  // To generate the id for finding block.
  uint64_t id_;
  std::map<std::string, Stat> stats_;
};

// to manage GMM worker
class gmm_worker {
  int cur_dev;
  int launcher_dev;
  pid_t pid;
  pid_t ppid;
  bool create_ctx;

  size_t init_dev_memPool_sz;
  size_t max_dev_memPool_sz;
  size_t dev_buf_sz;
  size_t cpu_buf_sz;
  size_t dev_free_sz;
  size_t dev_tot_sz;
  size_t borrow_sz;
  int test_sz;
  int config_fd;
  int lock_fd;
  int max_retry;

  char gmm_socket_path[MAX_SHM_PATH_LEN];
  char log_file[128];

  char *gpu_buf;
  char *gpu_buf2;
  char *cpu_buf;
  const char *ipc_dir;
  gmm_config_t *config;

  struct sockaddr_un addr;
  int socket_fd;
  int admin_connect;

  CUdevice cu_dev;
  CUcontext ctx;
  // evt for pre/post evt, support IPC
  CUevent mp_evt[MAX_DEV_NR][MAX_IPC_EVT_NR];
  // 0: in_stream(H2D), 1: out_stream(D2H), one stream(for bother DMA and
  // compute) per peer (exclude self)
  CUstream dm_stream[MAX_PATH_NR + 1];
  // ctrl evts shared for all dm_streams
  CUevent dm_evt[MAX_DM_EVT_NR];  // evt for DMA pipeline
  gmm_evt_queue free_dm_evt_pool;

  gmm_shm_table shmHost_table;
  gmm_shm_table shmDev_table;
  CUmemAccessDesc accessDesc;

  gmm_memPool *dev_memPool;  // pre-alloc dev mem pool
  gmm_vstore_mgr store_mgr;  // cached obj

 public:
  std::unique_ptr<TieringCache> tcache_;
  std::deque<UnitReq> scatter_1st_reqs_;
  std::deque<UnitReq> scatter_2nd_reqs_;
  std::deque<UnitReq> gather_1st_reqs_;
  std::deque<UnitReq> gather_2nd_reqs_;

  gmm_worker(worker_args *arg) {
    pid = getpid();
    ppid = arg->launcher_pid;
    create_ctx = arg->create_ctx;
    cur_dev = arg->cur_dev;
    launcher_dev = arg->launcher_dev;

    init_dev_memPool_sz = arg->init_dev_memPool_sz;
    max_dev_memPool_sz = arg->max_dev_memPool_sz;

    config_fd = 0;
    lock_fd = 0;
    socket_fd = 0;
    max_retry = 10;

    dev_buf_sz = cpu_buf_sz = 0;
    dev_free_sz = dev_tot_sz = borrow_sz = 0;
    test_sz = 2UL << 20;

    gpu_buf = gpu_buf2 = cpu_buf = nullptr;
    config = nullptr;
  }

  ~gmm_worker() {
    if (admin_connect > 0) {
      close(admin_connect);
    }

    // TODO
    if (cpu_buf) {
      free(cpu_buf);
    }
    if (gpu_buf) {
    }
    if (gpu_buf2) {
    }

    if (dev_memPool) delete dev_memPool;

    if (ctx) {
      // TODO delete other GPU resource
      CHECK_DRV(cuCtxDestroy(ctx))
    }
    // TODO
    if (config) {
      munmap(config, sizeof(gmm_config_t));
    }

    if (config_fd > 0) {
      close(config_fd);
    }

    if (lock_fd > 0) {
      close(lock_fd);
    }

    if (gmm_is_file_exist(gmm_socket_path)) {
      remove(gmm_socket_path);
    }

    if (gmm_log_file) {
      fclose(gmm_log_file);
    }
  }

  int get_socket() { return socket_fd; }
  gmm_config_t *get_config() { return config; }
  size_t get_gpuBuf_size() { return dev_buf_sz; }

  CUstream get_inStream() { return dm_stream[0]; }
  CUstream get_outStream() { return dm_stream[1]; }
  CUstream get_stream(int dev_idx) { return dm_stream[dev_idx + 2]; }
  char *get_gpuBuf() { return gpu_buf; }
  char *get_gpuBuf2() { return gpu_buf2; }

  int init();
  int init_dm_res();
  int register_and_serve();

  int get_evt_at(int dev_idx, uint32_t evt_idx, CUevent &pEvt) {
    if (dev_idx >= MAX_DEV_NR || evt_idx >= MAX_IPC_EVT_NR) {
      LOGGER(ERROR, "Invalid dev:%d or evt:%d", dev_idx, evt_idx);
      return -1;
    }

    if (pid != config->gmm_evt[dev_idx].creator_pid) {
      if (mp_evt[dev_idx][evt_idx] == 0) {
        CHECK_DRV(
            cuIpcOpenEventHandle(&mp_evt[dev_idx][evt_idx],
                                 config->gmm_evt[dev_idx].evt_handle[evt_idx]));
      }
      pEvt = mp_evt[dev_idx][evt_idx];
      LOGGER(DEBUG, "IPC evt at src dev:%d evt_idx:%d evt:%p IPC evt", dev_idx,
             evt_idx, pEvt);
    } else {
      pEvt = config->gmm_evt[dev_idx].evt[evt_idx];
      LOGGER(DEBUG, "evt at src dev:%d evt_idx:%d evt:%p same-process", dev_idx,
             evt_idx, pEvt);
    }

    return 0;
  }

  int prepare_evt(gmm_ipc_worker_req &req, CUevent &pre_evt,
                  CUevent &done_evt) {
    int ret = 0;

    if (req.gmm_op >= GMM_OP_DM_START_MARK &&
        req.gmm_op <= GMM_OP_DM_END_MARK) {
      get_evt_at(cur_dev, req.worker_evt_idx,
                 done_evt);  // must to sync to complete
      if (req.async) {
        get_evt_at(req.src_dev, req.pre_evt_idx, pre_evt);
      }
    }

    return ret;
  }

  int dm_handler(gmm_ipc_worker_req &req, CUevent &pre_evt, CUevent &post_evt);
  int dm_sg_handler(gmm_ipc_worker_req &req, CUevent &pre_evt,
                    CUevent &post_evt);
  int dm_cache_handler(gmm_ipc_worker_req &req, CUevent &pre_evt,
                       CUevent &post_evt, int conn_fd);

  int register_cuda_shm_handler(gmm_ipc_worker_req &req,
                                gmm_shmInfo_t *&shm_out, shared_fd fd);
  int register_cpu_shm_handler(gmm_ipc_worker_req &req,
                               gmm_shmInfo_t *&shm_out);

  inline int insert_sg_info(gmm_id gid, uint16_t slot, char *host_addr,
                            size_t bytes, size_t split_offset,
                            size_t base_offset, size_t cur_addr) {
    store_mgr.new_store(cur_dev, gid, slot, cur_dev, (uint64_t)host_addr, bytes,
                        split_offset, base_offset, cur_addr);
    return 0;
  }

  inline int find_sg_info(uint16_t slot, char *host_addr,
                          CUdeviceptr &cur_addr) {
    int ret = 1;
    std::vector<gmm_vstore_entry *> *entry =
        store_mgr.find_store(slot, (uint64_t)host_addr);
    if (entry) {
      // get cur_addr based on state
      // if (entry->at(0)->state == VSTORE_WORKER_SAVED ) {
      cur_addr = entry->at(0)->cur_addr;
      //}
      ret = 0;
    }

    return ret;
  }

  inline void delete_sg_info(uint16_t slot, char *host_addr) {
    store_mgr.delete_store(slot, (uint64_t)host_addr);
  }
};

gmm_state gmm_launch_worker_thr(int cur_dev, int worker_dev, bool create_ctx);
