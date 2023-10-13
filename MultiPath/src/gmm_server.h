#pragma once
#include <sys/mman.h>
#include <sys/wait.h>

#include <unordered_map>

#include "gmm_common.h"
#include "gmm_cuda_common.h"
#include "gmm_host_shm.h"
#include "gmm_queue.h"
#include "gmm_vstore.h"

struct gmm_stats_unit {
  size_t tot_req;
  size_t tot_req_bytes;

  size_t req_done;
  size_t req_bytes_done;
};

struct gmm_dm_stats {
  gmm_stats_unit stats;
  std::array<gmm_stats_unit, GMM_PRI_NUM> priority_stats;
};

// resource shall be reclaimed once client exits
struct client_tmp_resource {
  short slot_id;
  // TODO others evts ...

 public:
  client_tmp_resource(short slot_) : slot_id(slot_){};

 public:
  short get_slotID() const { return slot_id; }
};

class gmm_mgr_t {
 public:
  size_t tid;  // always inc
  gmm_stats_unit node_stats;
  std::array<gmm_dm_stats, MAX_DEV_NR> dev_stats;

  char *cpu_buffer;
  gmm_config_t *config_shm;
  int config_fd;
  int devCnt;
  uint64_t counter;  // always inc with new client

  // src_dev[] -- <pid, ...> -- <CUstream, ...> -- evtQ
  std::unordered_map<pid_t, std::unordered_map<CUstream, gmm_evt_queue>>
      busy_evt_list[MAX_DEV_NR];
  std::unordered_map<int, client_tmp_resource *> client_to_cleanup;

  gmm_evt_queue free_evt_list[MAX_DEV_NR];
  fifo_queue<uint16_t> slot_queue;

  gmm_vstore_mgr store_mgr;

 public:
  gmm_mgr_t(gmm_config_t *&config);

  ~gmm_mgr_t();

  size_t get_next_tid() { return ++tid; }

  // get/put req from global pool, also add
  int get_req(int cur_dev, gmm_req_t *req) { return 0; };

  int put_req(int cur_dev, gmm_req_t *req) {
    // TODO: drain req
    return 0;
  };

  int newDev_handler(int socket, gmm_ipc_admin_req &req);
  int delDev_handler(int socket, gmm_ipc_admin_req &req);

  int newClient_handler(int socket, gmm_ipc_admin_req &req,
                        gmm_ipc_admin_rsp &rsp);
  int delClient_handler(int socket);

  int choose_mp_handler(gmm_ipc_admin_req &req, gmm_ipc_admin_rsp &rsp);
  int choose_shm_neighbor_handler(gmm_ipc_admin_req &req,
                                  gmm_ipc_admin_rsp &rsp);

  int reclaim_evt_handler(gmm_ipc_admin_req &req);

  // reclaim resources from the given stream
  int reclaim_stream_handler(gmm_ipc_admin_req &req);

  // reclaim resources from the given client
  int reclaim_dev_handler(gmm_ipc_admin_req &req);

  void get_slot(uint16_t *idx) { *idx = *slot_queue.pop().get(); }

  void put_slot(uint16_t slot) { slot_queue.push(slot); }

  void print_stats(int dev = -2) {
    // TODO; print global or specific stats
  }
};

void *gmm_admin_proc(void *ready);
