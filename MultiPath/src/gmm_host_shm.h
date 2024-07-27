#pragma once

#include <errno.h>
#include <fcntl.h>
#include <memory.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <unordered_map>
#include <vector>

#include "gmm_common.h"

struct shm_peer_info {
  int dev_id;
  uint64_t shmInfo_addr;
};

enum gmm_ipc_mem_e {
  GMM_IPC_MEM_HOST,
  GMM_IPC_MEM_HOST_SHARE, /* worker that share same mem*/

  GMM_IPC_MEM_HOST_PIN,
  GMM_IPC_MEM_HOST_PIN_SHARE,

  GMM_IPC_MEM_NV_DEV,
  GMM_IPC_MEM_NV_DEV_SHARE,
};

struct gmm_shmInfo_t {
  gmm_ipc_mem_e type;  // 0: host_mem, 1: nv_gpu

  int dev_id;
  pid_t pid;
  uint32_t shm_idx;

  int shm_fd;
  CUmemGenericAllocationHandle vmm_handle;

  void *addr;  // host mem or CUdeviceptr(8B)
  size_t size;

  int shm_num;
  shm_peer_info shm_peer[MAX_PATH_NR];  // ptr to peer shmInfo_t, idx is 0,1,...

 public:
  // creator for cuda host pinned mem
  gmm_shmInfo_t(gmm_ipc_mem_e type_, int dev_id_, pid_t pid_, uint32_t idx_,
                size_t size_)
      : type(type_),
        dev_id(dev_id_),
        pid(pid_),
        shm_idx(idx_),
        shm_fd(0),
        addr(nullptr),
        size(size_),
        shm_num(0) {
    for (int i = 0; i < MAX_PATH_NR; ++i) {
      shm_peer[i].dev_id = -1;
      shm_peer[i].shmInfo_addr = 0ULL;
    }
  }

  gmm_shmInfo_t(gmm_ipc_mem_e type_, void *in_addr)
      : type(type_),
        dev_id(0),
        pid(getpid()),
        shm_idx(0),
        shm_fd(0),
        addr(nullptr),
        size(0),
        shm_num(0) {}

  // creator for cuMemAlloc
  gmm_shmInfo_t(gmm_ipc_mem_e type_, int dev_id_, void *in_addr,
                CUmemGenericAllocationHandle handle, size_t sz, int fd)
      : type(type_),
        dev_id(dev_id_),
        pid(getpid()),
        shm_idx(0),
        shm_fd(fd),
        vmm_handle(handle),
        addr(in_addr),
        size(sz),
        shm_num(0) {}

  ~gmm_shmInfo_t() {  // TODO
  }

  gmm_ipc_mem_e get_type() { return type; }
  int get_devID() { return dev_id; }
  pid_t get_pid() { return pid; }
  uint32_t get_idx() { return shm_idx; }
  size_t get_size() { return size; }
  void *get_addr() { return addr; }
  int get_shmFd() { return shm_fd; }
  int get_shmNum() { return shm_num; }

  CUmemGenericAllocationHandle get_handle() { return vmm_handle; }
  shm_peer_info get_peerShm(int idx) { return shm_peer[idx]; }

  uint64_t get_peerShm_addr(int dev_id) {
    int i = 0;
    for (; i < shm_num; ++i) {
      if (dev_id == shm_peer[i].dev_id) {
        return shm_peer[i].shmInfo_addr;
      }
    }
    return 0ULL;
  }
};

/*
struct gmm_shm_cmp {
  // desc order
  bool operator() (const gmm_shmInfo_t * left, const gmm_shmInfo_t * right) {
    return (left->addr  > right->addr);
  }
};
*/

struct gmm_shm_table {
  std::unordered_map<void *, gmm_shmInfo_t *&> shm_table;

 public:
  void add_shmEntry(gmm_shmInfo_t *&shm) {
    shm_table.emplace(shm->get_addr(), shm);
  }

  gmm_shmInfo_t *find_shmEntry(void *ptr) {
    auto ent = shm_table.find(ptr);
    if (ent != shm_table.end()) {
      return ent->second;
    } else {
      return nullptr;
    }
  }

  void del_shmEntry(void *ptr) {
    auto ent = shm_table.find(ptr);
    if (ent != shm_table.end()) {
      delete ent->second;
      shm_table.erase(ptr);
    }
  }
};

int gmm_shmCreate(const char *name, size_t sz, gmm_shmInfo_t *&info);
int gmm_shmOpen(const char *name, size_t sz, gmm_shmInfo_t *&info);
void gmm_shmClose(gmm_shmInfo_t *&info);

typedef pid_t Process;
int spawnProcess(Process *process, const char *app, char *const *args);
int waitProcess(Process *process);

#define checkIpcErrors(ipcFuncResult)                          \
  if (ipcFuncResult == -1) {                                   \
    fprintf(stderr, "Failure at %u %s\n", __LINE__, __FILE__); \
    exit(EXIT_FAILURE);                                        \
  }

struct ipcHandle_st {
  int socket;
  char *socketName;
};
typedef int ShareableHandle;

typedef struct ipcHandle_st ipcHandle;

int ipcCreateSocket(ipcHandle *&handle, const char *name,
                    const std::vector<Process> &processes);
int ipcOpenSocket(ipcHandle *&handle);
int ipcCloseSocket(ipcHandle *handle);
int ipcRecvShareableHandles(ipcHandle *handle,
                            std::vector<ShareableHandle> &shareableHandles);
int ipcSendShareableHandles(
    ipcHandle *handle, const std::vector<ShareableHandle> &shareableHandles,
    const std::vector<Process> &processes);
int ipcCloseShareableHandle(ShareableHandle shHandle);
