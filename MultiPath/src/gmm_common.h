// common settings, communictions btw client and server, worker
#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <semaphore.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <atomic>
#include <map>
#include <vector>

#include "gmm_cuda_common.h"
#include "gmm_util.h"

typedef uint64_t gmm_id;
const int GMM_ADMIN_PORT = 9573;
const uint32_t MAX_DEV_NR = 8;
const uint32_t MAX_IPC_EVT_NR = 1024;
const uint32_t MAX_PATH_NR = 8;
const uint32_t MAX_MP_PATH_NR = 8;
const uint32_t MAX_MP_STREAM_NR = 8;
const uint32_t MIX_MP_ENABLE_MB = 32;
const uint32_t MAX_SHM_PATH_LEN = 512;
const uint32_t MAX_DM_EVT_NR = 32;  // shared for all dm streams, finally 256?
const uint16_t GMM_SERVER_MAX_SLOT = 8192;  // 2^13

const uint32_t GMM_GDR_MAX_SZ_DEFAULT = 65536;

const int CPU_DEV = -1;
const int ANY_DEV = -2;
const int INVALID_DEV = -3;

const uint32_t MAX_REQ_NR = 1024 * 8;
const int MAX_CLIENT_NUM = 256;
const int MAX_PENDING_EVENT = 256;  // per client

const int MIN_LINK_WEIGHT = 25;
const size_t MIN_DM_ALIGNMENT = 512;  // Byte

// one per job/pod
static const char gmm_admin_socket[] = "gmm_admin.sock";
static const char GMM_CONFIG_SHM[] = "gmm_config";
static const char GMM_LOCK_FILE[] = "/var/run/gmm_lock";
static const char GMM_DEFAULT_IPC_DIR[] = "/tmp";
static const char GMM_HOST_SHM_PREFIX[] = "gmm_hShm";

const int SELF_LINK_PERF_RANK = -1;
enum LINK_PERF_RANK {
  NVLINK_PERF_DOUBLE = 0,
  NVLINK_PERF_SINGLE = 1,
  PCIEV3_PERF = 2,
};

enum DD_PATH_FLAG {
  DD_PATH_INVALID = -2,
  DD_PATH_P2P = -1,  // just P2P, no itermediate dev(>=0) needed
};

enum gmm_state {
  GMM_STATE_INIT = 0,

  GMM_STATE_ADMIN_READY,
  GMM_STATE_ADMIN_EXIST,
  GMM_STATE_ADMIN_ERROR,

  GMM_STATE_WORKER_READY,
  GMM_STATE_WORKER_EXIST,
  GMM_STATE_WORKER_ERROR,

  GMM_STATE_WORKER_DRAIN,
  GMM_STATE_WORKER_EXIT,
  GMM_STATE_ADMIN_EXIT,
};

enum gmm_worker_mode {
  GMM_MODE_GLOBAL =
      0, /* each worker is global unique process, worker create ctx */
  GMM_MODE_DEFAULT = 0,

  GMM_MODE_DDP_BIND, /* each worker is a thread launched by coresponding torch
                        DDP process, worker share ctx with client */
  GMM_MODE_DP_BIND, /* all workers are thread launched by a client processs, one
                       of workers share current ctx with client, other workers
                       create new ctx */
};

enum gmm_ipc_op {
  GMM_OP_INIT = 0,

  GMM_OP_NEW_WORKER = 1,
  GMM_OP_DEL_WORKER = 2,

  GMM_OP_NEW_CLIENT = 3,
  GMM_OP_DEL_CLIENT = 4,

  GMM_OP_ALLOC = 5,
  GMM_OP_FREE = 6,

  GMM_OP_PATH_QUERY = 7,
  GMM_OP_PATH_QUERY_SHM = 8,
  GMM_OP_RECLAIM_EVT = 9,
  GMM_OP_SYNC_STREAM = 10,
  GMM_OP_SYNC_DEV = 11,

  GMM_OP_REGISTER_SHM_HOST = 12,
  GMM_OP_DEREGISTER_SHM_HOST = 13,

  GMM_OP_REGISTER_SHM_DEV = 14,
  GMM_OP_DEREGISTER_SHM_DEV = 15,

  /*----------data moving op ------ */
  GMM_OP_DM_START_MARK, /* not use*/

  GMM_OP_H2D, /* don't change the order*/
  GMM_OP_D2H,
  GMM_OP_D2D,

  GMM_OP_SCATTER_PREPARE, /* to admin */
  GMM_OP_SCATTER,         /* to workers */
  GMM_OP_GATHER,          /*  to workers */

  GMM_OP_DM_END_MARK, /* not use*/
  /*----------data moving op ------ */

  GMM_OP_STOP,
  GMM_OP_INVALID,
};

enum gmm_req_state {
  GMM_STATE_COMPLETE = 0,
  GMM_STATE_INPROG = 1,
  GMM_STATE_INVALID = 2,
};

enum gmm_dm_priority {
  GMM_PRI_DEFAULT = 1,

  GMM_PRI_HIGH = 0,
  GMM_PRI_MEDIUM = 1,
  GMM_PRI_LOW = 2,

  GMM_PRI_NUM = 3,

  GMM_PRI_INVALID = 3,
};

enum gmm_dm_type {
  GMM_DM_KERNEL = 0,
  GMM_DM_DMA,
  GMM_DM_DMA_PIPELINE,
  GMM_DM_DMA_DIRECT,
};

// attribute for a given dm path
// orig_stream----record(pre_evt)                                wait(evt1/2),
// record(post) -> worker:                 |<-wait(pre)  ...
// record(worker_evt1)<--| worker:                 |<-wait(pre)  ...
// record(worker_evt2)<--|
struct gmm_path_option {
  int dev;
  pid_t creator_pid;
  gmm_dm_priority priority;
  gmm_dm_type type;

  // created by worker_dev(==req.cur_dev), assigned by admin(for specific dev)
  // recorded on orig_stream by client(noIPC), waited by worker_steram(may IPC)
  uint32_t pre_evt_idx;
  // created by worker_dev(==req.cur_dev), assigned by admin(for specific dev),
  // recorded on orig_stream by client(noIPC)
  uint32_t post_evt_idx;

  // created by worker, assigned by admin(for each dev),
  // recorded on worker_stream by worker-thread(no IPC), waited by
  // orig_stream(may IPC)
  uint32_t worker_evt_idx;
  size_t tmp_buf_addr;  // assigned a tmp addr for DMA; and SG
};

struct gmm_shm_neighbor {
  int path_nr;
  int path[MAX_PATH_NR];
};

struct gmm_path_query {
  gmm_id gid;
  int path_nr;

  // only one pre and post evt, created by worker_dev(== req.cur_dev), assigned
  // by admin,
  uint32_t pre_evt_idx;
  uint32_t post_evt_idx;
  gmm_path_option path[MAX_PATH_NR];
};

struct gmm_ipc_admin_req {
  gmm_ipc_op op;

  union {
    struct {
      pid_t pid;
      int dev_cnt;
    } newClient_req;

    struct {
      int dev_id;
      pid_t pid;
      char dev_bus[20];
    } newDev_req;

    struct {
      pid_t pid;
      int dev_id;
      char dev_bus[20];
    } register_shm_req;

    struct {
      pid_t pid;
    } attach_req;

    struct {
      pid_t pid;
    } detach_req;

    struct {
      int dev_id;       // gpu_id in H2D/D2H, or src_gpu_id in D2D
      int d2d_tgt_dev;  // set in D2D
      pid_t pid;
      size_t bytes;
      CUstream stream;
      gmm_ipc_op dm_type;
      bool async;
    } path_req;

    struct {
      int dev_id;
      pid_t pid;
      CUstream stream;
    } sync_req;

    gmm_path_query sync_mp;  // to reclaim its evt
  } data;

  gmm_ipc_admin_req() { memset(&data, 0, sizeof(data)); }
};

struct client_attach_rsp {
  uint16_t slot_id;
  uint64_t uuid;
};

struct gmm_ipc_admin_rsp {
  int status;

  union {
    client_attach_rsp new_client;

    gmm_path_query mp;

    gmm_shm_neighbor neighbors;
  } data;

  gmm_ipc_admin_rsp() { memset(&data, 0, sizeof(data)); }
};

// static setup based on topologies
struct mp_prefer {
  int mp_nr;
  int mp[MAX_PATH_NR];  // value: dev_idx
};

// req to worker
struct gmm_ipc_worker_req {
  gmm_ipc_op gmm_op;
  uint16_t slot_id;
  gmm_id gid;

  pid_t pid;    // client process
  int req_idx;  // to index req via shm across process

  int src_dev;     // data moving src
  int tgt_dev;     // data moving tgt
  int worker_dev;  // worker's dev

  //        |--------|**********||**********||**********||********
  // allocBase   base_offset    split_offset  split_offset
  //             src_addr/tgt_addr
  // at worker: mapped_baseAtWorker + map_offset + split_offset
  size_t base_offset_src;  // map_offset_src; // offset over alloc base addr for
                           // src data
  size_t base_offset_tgt;  // map_offset_tgt; // offset over alloc base addr for
                           // tgt data

  size_t split_offset_src;  // src_offset; //offset from src_addr
  size_t split_offset_tgt;  // tgt_offset; //offset from tgt_addr

  uint64_t shmInfo_addr_src;  // shmInfo addr for data src at worker
  uint64_t shmInfo_addr_tgt;

  uint32_t shm_idx;  // for host shared mem
  int shared_fd;     // for nv dev shared mem

  char *src_addr;  // src addr, 8B
  char *tgt_addr;
  char *gather_addr;
  size_t byte;

  uint32_t worker_evt_idx;
  uint32_t pre_evt_idx;
  gmm_dm_type dm_type;
  gmm_dm_priority priority;
  bool cross_process;
  bool async;
};

struct gmm_ipc_worker_rsp {
  int status;

  union {
    uint64_t shmInfo_addr;
  } data;

 public:
  gmm_ipc_worker_rsp() : status(0) { data.shmInfo_addr = 0UL; }
  gmm_ipc_worker_rsp(int ret, uint64_t addr) : status(ret) {
    data.shmInfo_addr = addr;
  }
};

class gmm_req_t {
 public:
  int req_idx;      // 0, ..., max_req
  uint64_t req_id;  // unique id, always inc
  int src_dev;
  int tgt_dev;
  void *src_addr;
  void *tgt_addr;
  void *gather_addr;
  size_t tot_size;
  int priority;  // input priority

  gmm_req_state state;
  gmm_ipc_op gmm_op;
  int task_num;

  // std::array<gmm_ipc_worker_req, MAX_DEV_NR> dm_sub_task;
  gmm_ipc_worker_req dm_task[MAX_PATH_NR];
  char dev_bus_id[20];  // src GPU dev
  pthread_mutex_t mutex;

  pid_t src_pid;

  int flag;
  CUresult err;
  sem_t req_sem;
  sem_t done_sem;
  size_t bytes;
  cudaIpcMemHandle_t ipc_mHandle;           // set when alloc
  CUdeviceptr ipc_dptr;                     // set when free
  CUmemGenericAllocationHandle vmm_handle;  // set when free
  int handle_fd;                            // sharable fd, set when alloc

 public:
  gmm_req_t(int id) : req_idx(id) {}  // used for pool
  gmm_req_t() : req_idx(0) {}
  ~gmm_req_t() {}

  // for H2D, D2H, D2D
  void set_param(char *tgt_a, int tgt_d, char *src_a, int src_d, size_t bytes,
                 int pri_in, gmm_ipc_op op);

  // block until req is done
  int synchronize();

  int query();
};

struct CmpKeyDesc {
  bool operator()(const int &k1, const int &k2) { return k1 <= k2; }
};

struct gmm_dev_evt_t {
  pid_t creator_pid;
  // a normal evt can support cross-device but doesn't support cross-process
  // evt with INTERPROCESS support cross-device and cross-process
  // CUevent evt_local[MAX_IPC_EVT_NR]; //pure local doesn't support IPC

  CUevent evt[MAX_IPC_EVT_NR];  // evt_local and evt_handle are pairs: handle is
                                // created from local
  CUipcEventHandle evt_handle[MAX_IPC_EVT_NR];
};

// global config and ctl info
// pre-alloc with large enough resource then shm with clients
// must with preknown size so as to mmap
struct gmm_config_t {
  char ipc_dir[512];

  size_t cpu_buf_sz;
  size_t dev_buf_sz;
  size_t min_mp_size;

  int tot_dev;
  int cuda_ver;
  uint16_t max_slot;  // max concurrent active client process;
  std::atomic<int> state;
  std::atomic<int> ready_cnt;
  pid_t creator_pid;
  bool remove_exist;
  bool sync;
  gmm_worker_mode worker_mode;

  // gmm_dev_evt_t gmm_evt[MAX_DEV_NR][MAX_DEV_NR];// for cross-process
  // [leader_gpu_idx][worker_gpu_idx]
  gmm_dev_evt_t gmm_evt[MAX_DEV_NR];  // for cross-process, one per dev

  char dev_bus[MAX_DEV_NR][20];
  int link_perf[MAX_DEV_NR][MAX_DEV_NR];
  int p2p_support[MAX_DEV_NR][MAX_DEV_NR];
  std::multimap<int, int, CmpKeyDesc>
      *dev_bw_perf_rank[MAX_DEV_NR];  // rank of perf: the lower #, the higher
                                      // perf: -1, 0, ..

  // std::vector<std::multimap<int, int, CmpKeyDesc>> dev_bw_perf_rank;
  gmm_req_t req_pool[MAX_REQ_NR];
  pid_t worker_creator[MAX_DEV_NR];

  // setup the prefer MP per static dev topologies
  int dev_cnt_per_PCIE_group;
  mp_prefer HD_MP[MAX_DEV_NR];
  mp_prefer SG_MP[MAX_DEV_NR];
  int DD_MP[MAX_DEV_NR][MAX_DEV_NR];

 public:
  gmm_config_t();
  void init();
  void set_state(int new_s) { state = new_s; };
  int get_state() { return state; };
  void get_bus_link(int cnt);
  void print_bus_link(int cnt);

  void setup_prefer_mp_HD();
  void setup_prefer_mp_DD();
  void setup_prefer_mp_SG();

  int choose_path_auto(gmm_ipc_admin_req &req, gmm_ipc_admin_rsp &rsp);

 private:
  bool has_slow_dev(int src_dev, int tgt_dev);
  bool same_PCIe_group(int src_dev, int tgt_dev);
};

int gmm_send_req(int connectID, gmm_req_t &gmm_req);
int gmm_recv_rsp(int connectFD, gmm_req_t &gmm_rsp, gmm_ipc_op op);

typedef int shared_fd;
ssize_t gmm_send(int connectFD, void *buffer, size_t bytes,
                 bool ctrl_msg = false, shared_fd fd = 0);
ssize_t gmm_recv(int connectFD, void *buffer, size_t bytes, int ctrl_flag = 0,
                 shared_fd *fd = nullptr);

gmm_state gmm_launch_admin_thr();
gmm_state gmm_launch_worker_thr(int launcher_dev, int worker_dev,
                                bool create_ctx);

// both client and server may call this
int choose_path(gmm_path_query &rsp, pid_t pid, gmm_ipc_op op,
                gmm_config_t *config);
int choose_path_auto(int src_d, gmm_path_query &rsp, pid_t pid, gmm_ipc_op op,
                     gmm_config_t *config);

static void gmm_enable_p2p(int cur_dev, int devCnt) {
  for (int j = 0; j < devCnt; j++) {
    if (j == cur_dev) continue;

    int ok = 0;
    CHECK_CUDA(cudaDeviceGetP2PAttribute(&ok, cudaDevP2PAttrAccessSupported,
                                         cur_dev, j));
    if (ok) {
      cudaDeviceEnablePeerAccess(j, 0);
    }
    cudaGetLastError();
  }
}
