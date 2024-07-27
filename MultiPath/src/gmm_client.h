#pragma once

#include <cuda.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <set>
#include <unordered_map>

#include "gmm_client_cfg.h"
#include "gmm_common.h"
#include "gmm_cuda_mem.h"
#include "gmm_gdr_plugin.h"
#include "gmm_host_mem.h"
#include "gmm_host_shm.h"
#include "gmm_queue.h"
#include "gmm_vstore.h"

struct gmm_req_t;
struct gmm_config_t;

struct gmm_lock_t {
  pthread_mutex_t mutex;
  pthread_mutexattr_t mutex_attr;
};

struct mp_status {
  std::atomic<bool> pending_mp;
};

struct shm_base_cmp {
  // desc order
  bool operator()(const void *left, const void *right) {
    return (left > right);
  }
};

enum mem_advise_t {
  MEM_ADVISE_NONE = 0,
  MEM_ADVISE_READONLY = 1, /* for data dedup */
};

struct shmDev_base_cmp {
  // desc order
  bool operator()(CUdeviceptr left, CUdeviceptr right) {
    return (left > right);
  }
};

// Global Mem Magt client: support global alloc, data moving (DM) with
// multi-path, global tiering one per client process
class gmm_client_ctx {
  uint64_t uuid;     // assigned by admin
  uint16_t slot_id;  // assigned by admin
  int dev_cnt;       // tot gpu num visible
  gmm_client_cfg *client_cfg;

  // GDR can offer much lower H2D latency when 1) size <=64KB, 2) host mem is
  // pinned
  int gdr_supported;
  gdr_t gdr_handle;

  size_t vmm_granularity;

  gmm_req_t *op_buf;  // to the shm area
  gmm_lock_t gmm_locks[MAX_DEV_NR];

  int admin_connect;                // cache the connect to gmm scheduler socket
  int worker_connects[MAX_DEV_NR];  // cache the connect to gmm worker socket

  // CUevent   mp_evt[MAX_DEV_NR][MAX_DEV_NR][MAX_IPC_EVT_NR];
  CUevent mp_evt[MAX_DEV_NR][MAX_IPC_EVT_NR];

  gmm_config_t *config_shm;
  int config_fd;
  int sched_fd;
  pid_t pid;
  fifo_queue<gmm_req_t *> req_pool;

  CUmemAllocationProp props[MAX_DEV_NR];
  CUmemAccessDesc accessDescs[MAX_DEV_NR];

  struct flock f_lock;

  // CUstream mp_stream[MAX_DEV_NR][3]; // H2D/D2H/D2D streams

  CUevent mp_event[MAX_DEV_NR]
                  [MAX_PENDING_EVENT];  // free list, shared for all client
                                        // threads and mp_stream

  std::mutex stream_map_lock;
  std::unordered_map<CUstream, bool> stream_map;
  std::unordered_map<CUevent, CUstream> evt_map;

  // TODO: merge and magt by single common set/map?
  // for host(pinned) mem allocation
  // std::set<void *, shm_base_cmp> hostMem_set; // set: desc ordered by dev
  // baseAddr
  std::map<void *, host_mem *> hostMem_map;  // map, key by dev baseAddr

  // for dev mem allocation
  // std::set<CUdeviceptr, shmDev_base_cmp> devMem_set; // set: desc ordered by
  // dev baseAddr
  std::map<CUdeviceptr, CUDA_devMem *> devMem_map;  // map, key by dev baseAddr

  // for pinned host shm
  std::atomic<uint32_t> shm_idx;
  // TODO: if assuming unified addressing, could use same key for both host and
  // dev memory
  // std::set<void *, shm_base_cmp> shm_baseAddr_set; // set: desc ordered by
  // host baseAddr
  std::map<void *, gmm_shmInfo_t *> shm_table;  // map key by addr

  // for each IPC-share dev mem, multi-gpu, large allocation
  // std::set<CUdeviceptr, shmDev_base_cmp> shmDev_baseAddr_set; // base addr
  // set: desc ordered by dev baseAddr std::unordered_map<CUdeviceptr,
  // gmm_shmInfo_t*> shmDev_table; // map key by dev baseAddr
  std::map<CUdeviceptr, gmm_shmInfo_t *>
      shmDev_table;  // map key by dev baseAddr

  gmm_vstore_mgr store_mgr;

 public:
  int get_devCnt() const { return dev_cnt; }
  bool get_gdrSupport() const { return (gdr_supported >= 1) ? true : false; }
  gdr_t get_gH() const { return gdr_handle; }
  uint16_t get_slotID() const { return slot_id; }
  uint64_t get_UUID() const { return uuid; }
  comb_addr get_combAddr(uint64_t addr) const {
    return get_comb_addr(slot_id, addr);
  }

 public:
  // run a set of pre functions
  int exec_preAlloc();
  int exec_alloc();
  int exec_postAlloc();

  int exec_preFree();
  int exec_free();
  int exec_postFree();

 public:
  gmm_client_ctx(gmm_client_cfg *&cfg);
  ~gmm_client_ctx();

  int fetch(char *tgt_addr, char *src_addr, size_t bytes, CUstream &stream);
  int evict(char *src_addr, size_t bytes, CUstream &stream);

  // DM via multi-path sync interface, no req handle needed
  int htod_async(char *tgt_addr, char *host_addr, size_t bytes,
                 CUstream &stream);

  inline int htod(char *tgt_addr, char *host_addr, size_t bytes) {
    CUstream stream = nullptr;
    return htod_async(tgt_addr, host_addr, bytes, stream);
  }

  inline int dtoh(char *host_addr, char *src_addr, size_t bytes) {
    CUstream stream = nullptr;
    return dtoh_async(host_addr, src_addr, bytes, stream);
  }

  inline int dtod(char *tgt_addr, char *src_addr, size_t bytes) {
    CUstream stream = nullptr;
    return dtod_async(tgt_addr, src_addr, bytes, stream);
  }

  // scatter-gather info
  inline int insert_sg_info(gmm_id gid, int dev_id, char *host_addr,
                            size_t bytes, gmm_ipc_worker_req *workers,
                            int worker_cnt) {
    store_mgr.new_store(gid, slot_id, dev_id, (uint64_t)host_addr, bytes,
                        workers, worker_cnt);
    return 0;
  }

  // dst_addr: the address copy to.
  // host_addr: search key.
  inline int find_sg_info(char *dst_addr, char *host_addr,
                          gmm_ipc_worker_req *workers, int &worker_cnt) {
    int ret = 1;
    std::vector<gmm_vstore_entry *> *entry =
        store_mgr.find_store(slot_id, (uint64_t)host_addr);
    if (entry) {
      worker_cnt = entry->size();
      for (int i = 0; i < worker_cnt; ++i) {
        workers[i].split_offset_src = entry->at(i)->get_splitOffset();
        workers[i].base_offset_src = entry->at(i)->get_baseOffset();
        workers[i].byte = entry->at(i)->get_rangeSize();
        workers[i].src_addr = host_addr + workers[i].split_offset_src;
        workers[i].tgt_addr = dst_addr + workers[i].split_offset_src;
        workers[i].worker_dev = entry->at(i)->worker_id;
      }
      ret = 0;
    } else {
      worker_cnt = 0;
    }
    return ret;
  }

  inline void delete_sg_info(char *host_addr) {
    store_mgr.delete_store(slot_id, (uint64_t)host_addr);
  }

  int dtoh_async(char *host_addr, char *src_addr, size_t bytes,
                 CUstream &stream);
  int dtod_async(char *tgt_addr, char *src_addr, size_t bytes,
                 CUstream &stream);

  // new interface: tiering data at src_addr to neighbor devs via multi-path
  int scatter(char *tgt_addr, char *src_addr, size_t bytes,
              const CUstream &stream);
  int gather(char *tgt_addr, char *src_addr, size_t bytes,
             const CUstream &stream);

  int scatter(char *src_addr, int src_d, size_t bytes, gmm_req_t *&req);
  int gather(char *tgt_addr, int tgt_d, size_t bytes, gmm_req_t *&req);

  int htod_async(char *tgt_addr, int tgt_d, char *host_addr, size_t bytes,
                 gmm_req_t *&req_out);
  int dtoh_async(char *host_addr, char *src_addr, int src_d, size_t bytes,
                 gmm_req_t *&req_out);
  int dtod_async(char *tgt_addr, int tgt_d, char *src_addr, int src_d,
                 size_t bytes, gmm_req_t *&req_out);

  // new API
  // e.g. RO thus able to dedup
  int malloc_hint(char *dptr, size_t bytes, const char *host_ptr,
                  mem_advise_t advise);

  // merge vector IOs at host with SIMD, then copy to dst, buf is optional
  // if buf provided, use that buf, else internally alloc and release imme
  // (sync) or lazy (async)
  int devMemCopyVectorHtoD(char *tgt, size_t bytes, void *vectors, int count,
                           char *buf);
  int devMemCopyVectorHtoD_async(char *tgt, size_t bytes, void *vectors,
                                 int count, char *buf);

  // query req status
  int query(int cur_dev, const gmm_req_t *&req);

  // blocking current thread until req_in done
  int synchronize(int cur_dev, gmm_req_t *req);

  // let stream wait on data moving for req
  int streamWait(int cur_dev, const cudaStream_t &stream, gmm_req_t *req);

  int reclaim_stream(const CUstream &stream, int dev_id);

  bool has_pending_mp(const CUstream &stream) {
    std::lock_guard<std::mutex> lock_(stream_map_lock);
    auto it = stream_map.find(stream);
    return (it != stream_map.end()) ? stream_map[stream] : false;
  }

  int reclaim_evt(CUevent &user_evt) {
    int ret = 0;

    auto it = evt_map.find(user_evt);
    if (it != evt_map.end()) {
      CUstream user_stream = evt_map[user_evt];
      if (has_pending_mp(user_stream)) {
        reclaim_stream(user_stream, -1);
        return ret;
      }
    }

    return ret;
  }

  void mark_evt(CUevent &evt, CUstream &stream) {
    if (has_pending_mp(stream)) evt_map.insert(std::make_pair(evt, stream));
  }
  pid_t get_pid() { return pid; }

  bool mp_ok(size_t bytes) {
    char *tmp_ =
        getenv("GMM_MP_OFF");  // client may set GMM_MP_OFF=1 to tmp disable MP
    return (config_shm && (bytes >= config_shm->min_mp_size) &&
            ((tmp_ ? atoi(tmp_) : 0) == 0))
               ? true
               : false;

    bool check = (config_shm && (bytes >= config_shm->min_mp_size));
    if (!check) {
      LOGGER(WARN, "input bytes:%ld failed to check", bytes);
    }
    return check;
  }

  // check after allocation
  bool gdr_ok(CUDA_devMem *&ent, host_mem *&host_ent, size_t current_io_bytes) {
    return (gdr_handle && ent->get_type() == GMM_MEM_TYPE_GDR &&
            (host_ent->get_type() == HOST_MEM_TYPE_PINNED ||
             host_ent->get_type() == HOST_MEM_TYPE_PINNED_SHM) &&
            current_io_bytes <= client_cfg->get_GDR_max_sz());
  }

  bool gdr_ok(size_t bytes) {
    // TODO: additional check on host addr ensure it's pinned memory
    return (gdr_handle && bytes <= client_cfg->get_GDR_max_sz());
  }

  bool init_and_set_gdr(gdr_t &gH) { return gmm_gdr_open(gH); }

  // check whether need to goto MP
  // -1: not necessary, as high BW exist
  // >=0: goto MP, the return val is the dev ID acting as inter-dev btw src and
  // tgt
  int dtod_mp_ok(char *tgt_addr, char *src_addr);

  uint32_t get_shmIdx() { return ++shm_idx; }

  void add_shmEntry(gmm_shmInfo_t *&shm, bool dev_type = false) {
    if (dev_type == false) {
      // shm_baseAddr_set.insert(shm->get_addr());
      // printf("--[%s] ptr:[%p %p]\n", __func__, shm->get_addr(),
      // (char*)shm->get_addr() + shm->get_size());
      shm_table[(char *)shm->get_addr() + shm->get_size()] = shm;
    } else {
      // printf("--[%s Dev] ptr:[%p %p]\n", __func__, shm->get_addr(),
      // (char*)shm->get_addr() + shm->get_size());
      // shmDev_baseAddr_set.insert((CUdeviceptr)shm->get_addr());
      shmDev_table[(CUdeviceptr)shm->get_addr() + shm->get_size()] = shm;
    }
  }

  void add_devMemEntry(CUDA_devMem *&ent) {
#if 0
    devMem_set.insert((CUdeviceptr)ent->get_addr());
    devMem_map[(CUdeviceptr)ent->get_addr()] = ent;
#endif
    devMem_map[(CUdeviceptr)ent->get_addr() + ent->get_orig_size()] = ent;
  }

  void add_hostMemEntry(host_mem *&ent) {
#if 0
    hostMem_set.insert(ent->get_addr());
    hostMem_map[ent->get_addr()] = ent;
#endif
    hostMem_map[(char *)ent->get_addr() + ent->get_orig_size()] = ent;
  }

  // first lookup baseAddr if found, then lookup shmTable via baseAddr to
  // further check range
  int get_shmInfo(void *ptr, gmm_shmInfo_t *&shmInfo, size_t *offset) {
    auto it = shm_table.upper_bound(ptr);
    if (it != shm_table.end()) {
      if (ptr >= (it->second->get_addr())) {
        *offset = ((char *)ptr) - (char *)(it->second->get_addr());
        shmInfo = it->second;
        return 0;
      }
    }
    // gtrace();
    LOGGER(WARN, "Faield to find dev shmInfo for dptr:%p", ptr);
    // printf("--Fail %s dptr:%p\n", __func__, ptr);
#if 0
    auto it = shm_baseAddr_set.lower_bound(ptr);

    if (it != shm_baseAddr_set.end()) {
      auto ent = shm_table.find(*it);
      if (ent != shm_table.end() && ptr >= ent->second->get_addr() && 
          ptr < ((char *)(ent->second->get_addr()) + ent->second->get_size())) {
        *offset = ((char *)ptr) - (char *)(ent->second->get_addr());
	shmInfo = ent->second;
	printf("--[%s] Found shm_set:%zu ptr:%p\n", __func__, shm_baseAddr_set.size(), ptr);
        return 0;
      }
    } else {
      LOGGER(WARN, "Faield to find shmInfo for ptr:%p", ptr);
      gtrace();
    }
#endif
    return 1;
  }

  int get_shmInfo_dev(CUdeviceptr dptr, gmm_shmInfo_t *&shmInfo,
                      size_t *offset) {
    auto it = shmDev_table.upper_bound(dptr);

    if (it != shmDev_table.end()) {
      if (dptr >= ((CUdeviceptr)it->second->get_addr())) {
        *offset = ((char *)dptr) - (char *)(it->second->get_addr());
        shmInfo = it->second;
        return 0;
      }
    }
    // gtrace();
    LOGGER(WARN, "Faield to find dev shmInfo for dptr:%llx", dptr);
    //printf("--Fail %s dptr:%p\n", __func__, dptr);
    return 1;
  }

  CUDA_devMem *find_devMemEntry(CUdeviceptr dptr) {
    auto it = devMem_map.upper_bound(dptr);
    if (it != devMem_map.end()) {
      if (dptr >= ((CUdeviceptr)it->second->get_addr())) {
        return it->second;
      }
    }
    // printf("--%s return nullptr\n", __func__);
    return nullptr;
#if 0
    auto ent = devMem_map.find(dptr);
    if (ent != devMem_map.end()) {
      return ent->second;
    } else {
      return nullptr;
    }
#endif
  }

  host_mem *find_hostMemEntry(void *ptr) {
    auto it = hostMem_map.upper_bound(ptr);
    if (it != hostMem_map.end()) {
      if (ptr >= (it->second->get_addr())) {
        return it->second;
      }
    }
    return nullptr;
#if 0
    auto ent = hostMem_map.find(ptr);
    if (ent != hostMem_map.end()) {
      return ent->second;
    } else {
      return nullptr;
    }
#endif
  }

  gmm_shmInfo_t *find_shmEntry(void *&base_ptr) {
    auto ent = shm_table.find(base_ptr);
    if (ent != shm_table.end()) {
      return ent->second;
    } else {
      return nullptr;
    }
  }

  gmm_shmInfo_t *find_shmDevEntry(CUdeviceptr dptr) {
    auto ent = shmDev_table.find(dptr);
    if (ent != shmDev_table.end()) {
      return ent->second;
    } else {
      return nullptr;
    }
  }

  void del_shmDevEntry(CUdeviceptr dptr) {
    auto ent = shmDev_table.find(dptr);
    if (ent != shmDev_table.end()) {
      delete ent->second;
      shmDev_table.erase(dptr);
    }

#if 0
    auto it = shmDev_baseAddr_set.find(dptr);
    if (it != shmDev_baseAddr_set.end()) {
      shmDev_baseAddr_set.erase(it);
    }
#endif
  }

  void del_devMemEntry(CUdeviceptr dptr) {
    auto ent = devMem_map.find(dptr);
    if (ent != devMem_map.end()) {
      delete ent->second;
      devMem_map.erase(dptr);
    }

#if 0
    auto it = devMem_set.find(dptr);
    if (it != devMem_set.end()) {
      devMem_set.erase(it);
    }
#endif
  }

  void del_hostMemEntry(void *&ptr) {
    auto ent = hostMem_map.find(ptr);
    if (ent != hostMem_map.end()) {
      delete ent->second;
      hostMem_map.erase(ptr);
    }

#if 0
    auto it = hostMem_set.find(ptr);
    if (it != hostMem_set.end()) {
      hostMem_set.erase(it);
    }
#endif
  }

  // delete from hashtable and baseAddr set
  void del_shmEntry(void *&ptr) {
    auto ent = shm_table.find(ptr);
    if (ent != shm_table.end()) {
      delete ent->second;
      shm_table.erase(ptr);
    }

#if 0
    auto it = shm_baseAddr_set.find(ptr);
    if (it != shm_baseAddr_set.end()) {
      shm_baseAddr_set.erase(it);
    }
#endif
  }

  int register_shm(int dev_id, gmm_shmInfo_t *&shm, bool dev_shm);
  int deregister_shm(gmm_shmInfo_t *&shm, bool dev_shm);

  // entry for pinned host mem alloc
  int hostMem_alloc(CUdevice cur_dev, void **&pp, size_t bytesize,
                    unsigned int Flags);
  // entry for pinned host mem free
  int hostMem_free(void *&p);

  // entry for dev mem allocation
  int devMem_alloc(CUdevice cur_dev, CUdeviceptr *&dptr, size_t bytesize);

  // entry for dev mem free
  int devMem_free(CUdevice cur_dev, CUdeviceptr dptr);

  int exec_devMem_preAlloc(CUdevice cur_dev, size_t bytesize);
  CUresult exec_devMem_alloc(CUdevice cur_dev, size_t bytesize,
                             CUDA_devMem *&ent);
  int exec_devMem_postAlloc(CUdevice cur_dev, CUDA_devMem *&ent);
  int devMem_postAlloc_export(CUdevice cur_dev, CUDA_devMem *&ent,
                              int &shared_fd);

  int exec_devMem_preFree(CUdevice cur_dev, CUdeviceptr dptr,
                          CUDA_devMem *&ent);
  CUresult exec_devMem_free(CUdevice cur_dev, CUdeviceptr dptr,
                            CUDA_devMem *&ent);
  int exec_devMem_postFree(CUdevice cur_dev, CUdeviceptr dptr,
                           CUDA_devMem *&ent);

 private:
  void gmm_close();
  bool is_ready();

  bool is_valid(int pid) {
    // User pid must not be 0 or 1.
    return pid > 1;
  }

  bool is_sameProcess() {
    bool same_process = true;

    for (int i = 0; i < MAX_DEV_NR; ++i) {
      int w_pid = config_shm->worker_creator[i];
      if (is_valid(w_pid) && pid != w_pid) {
        same_process = false;
      }
    }
    return same_process;
  }

  int get_evt_at(int launcher_dev, int dev_idx, uint32_t evt_idx,
                 CUevent &pEvt);
  void mark_pending_mp(const CUstream &stream);
  void reset_pending_mp(const CUstream &stream);
  int reclaim_evt(int dev_id, gmm_ipc_admin_req &mp);

  // connect to worker thr for the first time,then cache the connect
  void connect_if_not(int cur_dev, int tgt_dev);

  int client_send(int cur_dev, int tgt, void *buffer, size_t bytes);
  int client_recv(int tgt, void *buffer, size_t bytes);

  // perform multi-path data moving
  int mp_internal(char *tgt_addr, int tgt_d, char *src_addr, int src_d,
                  size_t bytes, gmm_req_t *&req_out, gmm_ipc_op op);
  int mp_dma_internal(char *tgt_addr, int tgt_d, char *src_addr, int src_d,
                      size_t bytes, const cudaStream_t &stream, gmm_ipc_op op);

  int set_req(gmm_req_t *req, int src_dev, int tgt_dev, size_t size,
              gmm_ipc_op op);

  // APIs for global mem magt
  /*
  void get_granularity(size_t *granularity, int dev_no);
  // dptr: use by client; ipc_dptr: ID used to free mem in neighbor
  cudaError_t alloc_neighbor(int tgt, int src, size_t bytes, CUdeviceptr &dptr,
CUdeviceptr &ipc_dptr);

  // API to free neighbor mem
  cudaError_t free_neighbor(int tgt, CUdeviceptr neighbor_dptr, void *dptr, int
src_dev = 0, size_t bytes = 0);

  int stop_neighbor(int tgt);
  void lock_neighbor(int tgt);
  void unlock_neighbor(int tgt);

// alloc GPU mem by sending req to GMM
// - GMM handle the alloc from proper neighbor GPU (enough free size and fast
inter-connect)
// - then map to local and return ptr
// input: size, tgt_dev, src_dev
// return: ret_ptr, shared_fd, vmm_handle
// - 0: ok and ret_ptr is updated
// - others: error
  CUresult gmm_alloc(CUdeviceptr &ret_ptr, size_t size, int tgt_dev, int
src_dev, int &shared_fd, CUmemGenericAllocationHandle &vmm_handle);

  // free memory: local first then notify neigbhor
  int gmm_free(CUdeviceptr client_dptr, size_t aligned_size, int src_dev, int
tgt_dev, int &shared_fd, CUmemGenericAllocationHandle &vmm_handle);
*/
};
