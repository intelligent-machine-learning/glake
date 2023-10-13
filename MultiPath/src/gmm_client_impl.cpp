#include <sys/socket.h>
#include <sys/un.h>

#include <chrono>

#include "gmm_api_stats.h"
#include "gmm_client.h"
#include "gmm_mp.h"

#if defined(MODULE_STATUS)
#undef MODULE_STATUS
#define MODULE_STATUS CUresult
#else
#define MODULE_STATUS CUresult
#endif

#define printf

// notice: use __CF("") to call CUDA directly, avoiding invoke hook again
extern void *libP;
// static thread_local FILE *gmm_log_file = nullptr;

static int gmm_send_fd(int connectFD, int src_dev, gmm_ipc_worker_req &req) {
  int ret = 0;
  struct msghdr msg;
  struct iovec iov;
  struct cmsghdr *cmsg;
  char cmsg_buf[CMSG_SPACE(sizeof(req.shared_fd))];

  memset(&msg, 0, sizeof(msg));
  msg.msg_control = cmsg_buf;
  msg.msg_controllen = sizeof(cmsg_buf);
  // send the file descriptor via SCM_RIGHTS
  cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_level = SOL_SOCKET;

  cmsg->cmsg_len = CMSG_LEN(sizeof(req.shared_fd));
  memcpy(CMSG_DATA(cmsg), &req.shared_fd, sizeof(req.shared_fd));

  iov.iov_base = &req;
  iov.iov_len = sizeof(req);
  // iov.iov_base = (void *)&src_dev; iov.iov_len = sizeof(src_dev);
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;

  if ((ret = sendmsg(connectFD, &msg, MSG_NOSIGNAL)) < 0) {
    LOGGER(ERROR, "send rsp failed, connectFD:%d error:%s", connectFD,
           strerror(errno));
  }

  return ret;
}

static int gmm_split_data(int cur_dev, pid_t pid, char *dst_addr,
                          char *src_addr, size_t bytes,
                          int link_perf[][MAX_DEV_NR], gmm_ipc_admin_rsp *mp,
                          gmm_ipc_worker_req *split_task, bool async) {
  int ret = 0;

  if (!src_addr || bytes == 0) {
    printf("-- src:%p, btytes:%zu\n", src_addr, bytes);
    abort();
    LOGGER(ERROR, "src:%p dst:%p bytes:%ld invalid\n", src_addr, dst_addr,
           bytes);
    return -1;
  }
  int path_nr = mp->data.mp.path_nr;
  size_t gran = NUM_LOOP_UNROLL * NUM_THREADS_IN_BLOCK *
                sizeof(ulong2);  // at least 4096B
  // size_t gran = (1<<20UL); // at least 4096B
  size_t left = bytes;
  char *src_tail = src_addr + bytes;
  char *dst_tail = dst_addr + bytes;

  int tot_weight = 0, w = 0;
  int path_weight[path_nr];
  for (int i = 0; i < path_nr; i++) {
    w = link_perf[cur_dev][mp->data.mp.path[i].dev];
    // TODO: Work around for SwapOut SwapIn benchmark.
    // if (0 ==mp->data.mp.path[i].dev) {
    //  w = 30;
    //}
    // printf("--Split: i=%d cur_dev:%d dst_dev:%d weight:%d\n", i, cur_dev,
    // mp->data.mp.path[i].dev, w);
    path_weight[i] = w;
    tot_weight += w;
  }

  size_t range_sz = 0;  //(bytes/path_nr) / gran  * gran;

  // split data based on weight of link path (the higher BW, the more data
  // assigned) , from addr tail to head, let the first path take all the left
  // TODO: consider additional factors like busy status
  gmm_ipc_worker_req *p = nullptr;
  for (int i = path_nr - 1; i >= 0; i--) {
    p = &split_task[i];
    range_sz = (path_weight[i] * bytes / tot_weight) / gran * gran;
    p->split_offset_tgt = 0;

    if (i == 0) {
      p->src_addr = src_addr;
      p->tgt_addr = dst_addr;
      p->byte = left;
      if ((src_tail != src_addr + left) || (dst_tail != dst_addr + left)) {
        LOGGER(INFO, "src_tail:%p vs %p, dst_tail:%p vs %p Not matched !",
               src_tail, src_addr + left, dst_tail, dst_addr + left);
      }

      p->split_offset_src = 0;
    } else {
      p->src_addr = src_tail - range_sz;
      p->tgt_addr = dst_tail - range_sz;
      p->byte = range_sz;
      p->split_offset_src = p->src_addr - src_addr;
    }

    src_tail -= range_sz;
    dst_tail -= range_sz;
    left -= range_sz;
    // printf("-- Split: i=%d src:%p dst:%p size:%zu KB, range_sz:%zu KB,
    // left:%zu KB\n",
    //    i, p->src_addr, p->tgt_addr, p->byte>>10UL, range_sz>>10UL,
    //    left>>10UL);

    p->src_dev = cur_dev;
    p->tgt_dev = mp->data.mp.path[i].dev;

    p->worker_evt_idx = mp->data.mp.path[i].worker_evt_idx;
    p->worker_dev = mp->data.mp.path[i].dev;
    p->async = async;
    p->pre_evt_idx = async ? mp->data.mp.pre_evt_idx : MAX_IPC_EVT_NR;
    p->dm_type = mp->data.mp.path[i].type;
    p->priority = mp->data.mp.path[i].priority;
    p->pid = pid;
    p->cross_process = !(mp->data.mp.path[i].creator_pid == pid);

    LOGGER(INFO,
           "Split into %d ranges path:%d size:%ld src:%p offset:%ld tgt:%p "
           "offset:%ld cross:%d",
           path_nr, i, range_sz, p->src_addr, p->split_offset_src, p->tgt_addr,
           p->split_offset_tgt, p->cross_process);
  }
  return ret;
}

gmm_client_ctx::gmm_client_ctx(gmm_client_cfg *&client_cfg_) {
  pid = getpid();
  printf("-- pid:%d\n", pid);
  sleep(5);
  slot_id = -1;
  uuid = 0;
  int cur_dev = 0;
  gmm_state status = GMM_STATE_INIT;

  CHECK_CUDA(cudaGetDevice(&cur_dev));
  CHECK_CUDA(cudaGetDeviceCount(&dev_cnt));

  // char worker_log_file[128];
  // snprintf(worker_log_file, 127, "/tmp/gmm-client-%d-%d.log", pid, cur_dev);
  // gmm_log_file = fopen(worker_log_file, "w");

  // gmm_set_log_level();
  client_cfg = client_cfg_;
  LOGGER(INFO,
         "To init gmm_client_ctx cur_dev:%d dev_cnt:%d tid:%d pid:%d --> P2P",
         cur_dev, dev_cnt, gettid(), pid);

  // try to enable GDR (if suported) for the first time
  gdr_supported = false;
  bool gdr_enable =
      getenv("GMM_GDR_ENABLE") ? atoi(getenv("GMM_GDR_ENABLE")) : false;
  if (gdr_enable) {
    gdr_supported = init_and_set_gdr(gdr_handle);
    LOGI("Checking GDR : %d", gdr_supported);
  }

  // gmm_enable_p2p(cur_dev, dev_cnt); // create ctx on other dev! and not
  // necessary for VMM, but required for IPC evt!! (why?)
  // CHECK_CUDA(cudaSetDevice(cur_dev));

  for (int tgt_dev = 0; tgt_dev < dev_cnt; tgt_dev++) {
    worker_connects[tgt_dev] = -1;
  }

  // 1. try to start admin thread if not ready
  // but always one single admin in a process-group (such as a container) no
  // matter DP/DDP
  if (getenv("WORLD_SIZE") ||
      (getenv("GMM_DDP") ? atoi(getenv("GMM_DDP")) : 0) ||
      (getenv("GMM_DP") ? atoi(getenv("GMM_DP")) : 0)) {
    status = gmm_launch_admin_thr();
    if (status == GMM_STATE_ADMIN_READY || status == GMM_STATE_ADMIN_EXIST) {
      LOGGER(INFO, "pid:%d cur_dev:%d GMM admin %s", pid, cur_dev,
             (status == GMM_STATE_ADMIN_READY) ? "created" : "already exists");
    } else {
      LOGGER(ERROR, "GMM admin launch return error:%d", status);
    }
    CHECK_CUDA(cudaSetDevice(cur_dev));
  }

  config_fd = shm_open(GMM_CONFIG_SHM, O_RDWR, 0666);
  // fprintf(stderr, "-- shm:%s fd:%d\n", GMM_CONFIG_SHM, config_fd);
  if (config_fd < 0) {
    perror("-- shm_open:");
    LOGGER(ERROR, "pid:%d cur_dev:%d error open %s error:%s\n", pid, cur_dev,
           GMM_CONFIG_SHM, strerror(errno));
    ASSERT(0, "Failed on shm_open");
  }

  config_shm =
      (gmm_config_t *)mmap(NULL, sizeof(gmm_config_t), PROT_READ | PROT_WRITE,
                           MAP_SHARED, config_fd, 0);
  if (config_shm == MAP_FAILED) {
    perror("-- mmap:");
    LOGGER(ERROR, "pid:%d cur_dev:%d failed to mmap %s error:%s\n", pid,
           cur_dev, GMM_CONFIG_SHM, strerror(errno));
    close(config_fd);
  }

  // 2. start worker threads if not default global mode
  if (config_shm->worker_mode > GMM_MODE_DEFAULT) {
    // start worker thread for cur_dev
    LOGGER(DEBUG, "pid:%d cur_dev:%d mode:%d, launching my worker ...", pid,
           cur_dev, config_shm->worker_mode);
    printf("pid:%d cur_dev:%d mode:%d, launching my worker ...\n", pid, cur_dev,
           config_shm->worker_mode);
    status = gmm_launch_worker_thr(cur_dev, cur_dev, true);
    if (status == GMM_STATE_ADMIN_READY) {
      LOGGER(DEBUG, "Launching worker done ...");
    }

    CHECK_CUDA(cudaSetDevice(cur_dev));

    if (config_shm->worker_mode == GMM_MODE_DP_BIND) {
      for (int i = 0; i < dev_cnt; ++i) {
        if (i == cur_dev) continue;
        LOGGER(DEBUG,
               "pid:%d cur_dev:%d DP mode, launching other workers %d ...", pid,
               cur_dev, i);
        status = gmm_launch_worker_thr(cur_dev, i, true);
      }
    }
    CHECK_CUDA(cudaSetDevice(cur_dev));
  }

  // 3. connect to admin and worker thr
  {
    struct sockaddr_un addr;
    char gmm_ipc_socket_path[MAX_SHM_PATH_LEN];
    admin_connect = 0;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
    snprintf(gmm_ipc_socket_path, sizeof(gmm_ipc_socket_path) - 1, "%s/%s",
             config_shm->ipc_dir, gmm_admin_socket);
#pragma GCC diagnostic pop
    admin_connect = socket(AF_UNIX, SOCK_STREAM, 0);
    if (admin_connect < 0) {
      LOGGER(ERROR, "pid:%d cur_dev:%d failed to create socket error:%s\n", pid,
             cur_dev, strerror(errno));
    }

    strncpy(addr.sun_path, gmm_ipc_socket_path, sizeof(addr.sun_path) - 1);
    addr.sun_family = AF_UNIX;

    if (connect(admin_connect, (struct sockaddr *)&addr,
                sizeof(struct sockaddr_un)) != 0) {
      LOGGER(ERROR, "pid:%d cur_dev:%d failed to connect to admin %s, error:%s",
             pid, cur_dev, gmm_ipc_socket_path, strerror(errno));
      ASSERT(0, "Failed on connect");
    }

    gmm_ipc_admin_req req;
    gmm_ipc_admin_rsp rsp;
    req.data.newClient_req.pid = pid;
    req.data.newClient_req.dev_cnt = dev_cnt;
    req.op = GMM_OP_NEW_CLIENT;

    if (gmm_send(admin_connect, (void *)&req, sizeof(req)) > 0 &&
        gmm_recv(admin_connect, (void *)&rsp, sizeof(rsp)) > 0) {
      LOGGER(INFO, "clinet slot_id:%d", rsp.data.new_client.slot_id);
      slot_id = rsp.data.new_client.slot_id;
      uuid = rsp.data.new_client.uuid;
    } else {
      LOGGER(ERROR, "pid:%d client failed to register to GMM admin", pid);
    }
  }

  // TODO: lock?
  if (config_shm->worker_mode == GMM_MODE_DDP_BIND) {
    while (config_shm->ready_cnt < dev_cnt) {
      sched_yield();
    }
  }
  LOGGER(DEBUG, "pid:%d cur_dev:%d all workers are ready to connect", pid,
         cur_dev);

  for (int tgt_dev = 0; tgt_dev < dev_cnt; ++tgt_dev) {
    connect_if_not(cur_dev, tgt_dev);
  }

  LOGGER(DEBUG, "pid:%d cur_dev:%d connect to GMM workers done", pid, cur_dev);

  for (int i = 0; i < MAX_REQ_NR; i++) {
    // req_pool.push(new gmm_req_t(i));
    // TODO: now only one client that take over all req, to support multi-client
    // that share req
    req_pool.push(&config_shm->req_pool[i]);
  }

  if ((sched_fd = open(GMM_LOCK_FILE, O_RDWR)) < 0) {
    LOGGER(ERROR, "failed to open %s error:%s", GMM_LOCK_FILE, strerror(errno));
  }

  CHECK_CUDA(cudaSetDevice(cur_dev));
  shm_idx = 1;
  LOGGER(INFO, "gmm client init done for pid:%d cur_dev:%d tot_dev:%d", pid,
         cur_dev, dev_cnt);
  printf("gmm client init done for pid:%d cur_dev:%d tot_dev:%d\n", pid,
         cur_dev, dev_cnt);
}

// detr
gmm_client_ctx::~gmm_client_ctx() {
  int cur_dev = 0;
  CHECK_CUDA(cudaGetDevice(&cur_dev));

  for (int d = 0; d < dev_cnt; ++d) {
    if (worker_connects[d] > 0) {
      LOGGER(VERBOSE, "Closing connect to %d id:%d", d, worker_connects[d]);
      close(worker_connects[d]);
    }
  }

  CHECK_CUDA(cudaSetDevice(cur_dev));
  LOGGER(INFO, "GMM client pid:%d exit\n", pid);

  if (gmm_log_file) {
    fclose(gmm_log_file);
  }
}

bool gmm_client_ctx::is_ready() {
  if (sched_fd > 0 && config_shm) {
    int ready_cnt = 0;
    f_lock.l_type = F_RDLCK;
    f_lock.l_whence = SEEK_SET;
    f_lock.l_start = 0;
    f_lock.l_len = 0;

    fcntl(sched_fd, F_SETLKW, &f_lock);
    ready_cnt = config_shm->ready_cnt;
    f_lock.l_type = F_UNLCK;
    fcntl(sched_fd, F_SETLKW, &f_lock);
    return (ready_cnt == config_shm->tot_dev);
  } else {
    return false;
  }
}

void gmm_client_ctx::gmm_close() {
  for (int i = 0; i < dev_cnt; i++) {
    if (worker_connects[i] > 0) {
      close(worker_connects[i]);
    }
  }
  for (int i = 0; i < MAX_REQ_NR; i++) {
    // TODO: now only one client that take over all req, to support multi-client
    // that share req
    std::shared_ptr<gmm_req_t *> obj = req_pool.pop();
    delete obj.get();
  }
  if (sched_fd > 0) close(sched_fd);
}

// client to get the usable evt
inline int gmm_client_ctx::get_evt_at(int launcher_dev_unused, int worker_dev,
                                      uint32_t evt_idx, CUevent &pEvt) {
  if (launcher_dev_unused >= MAX_DEV_NR || worker_dev >= MAX_DEV_NR ||
      evt_idx >= MAX_IPC_EVT_NR) {
    LOGGER(ERROR, "Invalid launcher_dev:%d or worker_dev:%d or evt:%d",
           launcher_dev_unused, worker_dev, evt_idx);
    return -1;
  }

  // if (pid !=
  // config_shm->gmm_evt[launcher_dev_unused][worker_dev].creator_pid) {
  if (pid != config_shm->gmm_evt[worker_dev].creator_pid) {
    pEvt = mp_evt[worker_dev][evt_idx];
  } else {
    pEvt = config_shm->gmm_evt[worker_dev].evt[evt_idx];
  }

  return 0;
}

// connect to worker thr for the first time,then cache the connect
// TODO: DDP check worker started
void gmm_client_ctx::connect_if_not(int cur_dev, int tgt_dev) {
  if (worker_connects[tgt_dev] <= 0) {
    char gmm_ipc_socket_path[MAX_SHM_PATH_LEN];
    struct sockaddr_un addr;
    pid_t worker_pid = config_shm->worker_creator[tgt_dev];

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
    snprintf(gmm_ipc_socket_path, sizeof(gmm_ipc_socket_path) - 1,
             "%s/gmm_worker_%d_%d.sock", config_shm->ipc_dir, worker_pid,
             tgt_dev);
#pragma GCC diagnostic pop

    worker_connects[tgt_dev] = socket(AF_UNIX, SOCK_STREAM, 0);
    if (worker_connects[tgt_dev] < 0) {
      LOGGER(ERROR, "pid:%d failed on create socket for %s, error:%s", pid,
             gmm_ipc_socket_path, strerror(errno));
    }

    strncpy(addr.sun_path, gmm_ipc_socket_path, sizeof(addr.sun_path) - 1);
    addr.sun_family = AF_UNIX;

    if (connect(worker_connects[tgt_dev], (struct sockaddr *)&addr,
                sizeof(struct sockaddr_un)) != 0) {
      LOGGER(ERROR, "pid:%d failed on connect to %s, error:%s", pid,
             gmm_ipc_socket_path, strerror(errno));
    }

    // IPC evt only for cross-process, otherwise error code=201
    // cudaErrorDeviceUninitialized
    if (pid != config_shm->gmm_evt[tgt_dev].creator_pid) {
      for (int i = 0; i < MAX_IPC_EVT_NR; i++) {
        CHECK_DRV(__CF("cuIpcOpenEventHandle")(
            &mp_evt[tgt_dev][i], config_shm->gmm_evt[tgt_dev].evt_handle[i]));
      }
      // LOGGER(DEBUG, "pid:%d cross-process use IPC evt", pid);
    }  // else: they're from the same process, use config_shm-
  }
}

int gmm_client_ctx::client_send(int cur_dev, int tgt, void *buffer,
                                size_t bytes) {
  int ret = 0;
  connect_if_not(cur_dev, tgt);
  if ((ret = gmm_send(worker_connects[tgt], buffer, bytes)) < 0) {
    LOGGER(ERROR, "pid:%d send req failed, tgt:%d error:%s", pid, tgt,
           strerror(errno));
  }
  return ret;
}

int gmm_client_ctx::client_recv(int tgt, void *buffer, size_t bytes) {
  int ret = 0;
  if ((ret = gmm_recv(worker_connects[tgt], buffer, bytes)) < 0) {
    LOGGER(ERROR, "pid:%d recv req failed, tgt:%d error:%s", pid, tgt,
           strerror(errno));
  }
  return ret;
}

// perform multi-path data moving
int gmm_client_ctx::mp_internal(char *tgt_addr, int tgt_d, char *src_addr,
                                int src_d, size_t bytes, gmm_req_t *&req_out,
                                gmm_ipc_op op) {
  return 0;
}

// Perform MP data moving via DMA
// - sync if stream is nullptr
// - note, running in client current thread, so doesn't support kernel
// TODO: code re-org: each op define logics like pre_handler(), post_handler().
// here we just orch those handlers, with common event/sync
int gmm_client_ctx::mp_dma_internal(char *tgt_addr, int tgt_d, char *src_addr,
                                    int src_d, size_t bytes,
                                    const CUstream &orig_stream,
                                    gmm_ipc_op op) {
  int ret = 0;
  int cur_dev = 0;
  int task_num = 1;
  int priority_req = 0;
  size_t base_offset_src = 0;
  size_t base_offset_tgt = 0;
  gmm_shmInfo_t *shm_src = nullptr;
  gmm_shmInfo_t *shm_tgt = nullptr;
  // printf("--[%s] Req src:%p dst:%p size:%zu MB\n", __func__, src_addr,
  // tgt_addr, bytes>>20UL);

  if (op == GMM_OP_H2D) {
    ret = get_shmInfo(src_addr, shm_src, &base_offset_src);
    if (ret != 0) {
      goto FAST_RETURN;
    }
    ret = get_shmInfo_dev((CUdeviceptr)tgt_addr, shm_tgt, &base_offset_tgt);
    if (ret != 0) {
      goto FAST_RETURN;
    }
    if (ret) {
      printf("-- [%s] H2D get_shmInfo_dev Fail, dev_ptr:%p\n", __func__,
             tgt_addr);
    } else {
      // printf("-- [%s] H2D get_shmInfo_dev OK, dev_ptr:%p\n", __func__,
      // tgt_addr);
    }
  } else if (op == GMM_OP_D2H) {
    ret = get_shmInfo_dev((CUdeviceptr)src_addr, shm_src, &base_offset_src);
    if (ret != 0) {
      goto FAST_RETURN;
    }
    ret = get_shmInfo(tgt_addr, shm_tgt, &base_offset_tgt);
    if (ret != 0) {
      goto FAST_RETURN;
    }
  } else if (op == GMM_OP_D2D) {
    ret = get_shmInfo_dev((CUdeviceptr)src_addr, shm_src, &base_offset_src);
    if (ret != 0) {
      goto FAST_RETURN;
    }
    ret = get_shmInfo_dev((CUdeviceptr)tgt_addr, shm_tgt, &base_offset_tgt);
    if (ret != 0) {
      goto FAST_RETURN;
    }
  }
  // printf("--[%s] ret:%d\n", __func__, ret);

  if (ret != 0) {
  FAST_RETURN:
    LOGGER(INFO,
           "shmInfo check failed:%d src_d:%d src_addr:%p tgt_d:%d tgt_addr:%p "
           "op:%d",
           ret, src_d, src_addr, tgt_d, tgt_addr, op);
    return ret;
  }

  CHECK_CUDA(cudaGetDevice(&cur_dev));
  gmm_ipc_worker_req dm_task[MAX_PATH_NR];
  for (int i = 0; i < MAX_PATH_NR; ++i) {
    dm_task[i] = {};
  }
  gmm_ipc_admin_req req = {};
  gmm_ipc_admin_rsp rsp = {};

  auto t1 = std::chrono::system_clock::now();

  // TODO: cache prev decision for N req w/o IPC communication
  if (op != GMM_OP_GATHER) {
    // 1.query admin to get multi-path
    req.data.path_req.pid = pid;
    req.data.path_req.dev_id = cur_dev;
    req.data.path_req.d2d_tgt_dev = tgt_d;
    req.data.path_req.bytes = bytes;
    req.data.path_req.stream = orig_stream;
    req.data.path_req.dm_type = op;

    if (op == GMM_OP_SCATTER)
      req.data.path_req.dm_type = GMM_OP_SCATTER_PREPARE;
    req.data.path_req.async = (orig_stream) ? true : false;
    ;

    req.op = GMM_OP_PATH_QUERY;
    // TODO: may just locking and update some shared struct rather than IPC
    if (gmm_send(admin_connect, (void *)&req, sizeof(req)) > 0 &&
        gmm_recv(admin_connect, (void *)&rsp, sizeof(rsp)) > 0) {
      // 2.split data ranges for each path
      task_num = rsp.data.mp.path_nr;
      ret = gmm_split_data(cur_dev, pid, tgt_addr, src_addr, bytes,
                           config_shm->link_perf, &rsp, &dm_task[0],
                           orig_stream ? true : false);
    } else {
      LOGGER(WARN,
             "GMM admin failed to get MP src_d:%d src_addr:%p tgt_d:%d "
             "tgt_addr:%p op:%d",
             src_d, src_addr, tgt_d, tgt_addr, op);
      return -1;
    }
  } else {
    find_sg_info(tgt_addr, src_addr, &dm_task[0], task_num);
  }

  if (task_num == 0) {
    LOGGER(WARN,
           "GMM mp path 0, fallback to normal, src_d:%d src_addr:%p tgt_d:%d "
           "tgt_addr:%p op:%d",
           src_d, src_addr, tgt_d, tgt_addr, op);
    return -1;
  }

  gmm_ipc_worker_req *task = &dm_task[0];
  CUevent pre_evt = nullptr, post_evt = nullptr, mp_evt[MAX_MP_PATH_NR];

  // 3. pre-evt on cur_dev's orig_stream
  if (orig_stream) {
    ret = get_evt_at(cur_dev, cur_dev, rsp.data.mp.pre_evt_idx, pre_evt);
    LOGGER(DEBUG, "pre-evt idx:%d cur_dev:%d pre_evt:%p stream:%p",
           rsp.data.mp.pre_evt_idx, cur_dev, pre_evt, orig_stream);
    CHECK_DRV(__CF("cuEventRecord")(pre_evt, orig_stream));
    LOGGER(DEBUG, "pre-evt idx:%d cur_dev:%d pre_evt:%p stream:%p done",
           rsp.data.mp.pre_evt_idx, cur_dev, pre_evt, orig_stream);
  }

  //////////////// multi-path
  // printf("-- task num:%d\n", task_num);
  for (int i = 0; i < task_num; ++i) {
    int gpu_id = task[i].worker_dev;
    task[i].slot_id = slot_id;
    task[i].gid = rsp.data.mp.gid;
    ret = get_evt_at(cur_dev, gpu_id, task[i].worker_evt_idx, mp_evt[i]);
    task[i].gmm_op = op;

    if (op == GMM_OP_H2D || op == GMM_OP_D2H || op == GMM_OP_D2D) {
      task[i].base_offset_src = base_offset_src;
      task[i].base_offset_tgt = base_offset_src;

      task[i].shmInfo_addr_src = shm_src->get_peerShm_addr(gpu_id);
      task[i].shmInfo_addr_tgt = shm_tgt->get_peerShm_addr(gpu_id);
    }

    // 3. submit to worker threads
    if ((op == GMM_OP_D2H) && task[i].dm_type > GMM_DM_KERNEL) {
      // for D2H DMA, NV GPU only supports cur_addr write peer_addr, not read,
      // hence have to delegate 'cur_dev' to handle pipeline/DMA via
      // multi-stream
      //  TODO: another impl option is to handle in client thread rather than
      //  worker?
      if (gmm_send(worker_connects[cur_dev], (void *)&task[i],
                   sizeof(task[0])) > 0 &&
          gmm_recv(worker_connects[cur_dev], (void *)&ret, sizeof(ret)) > 0) {
      }
    } else {
      if (gmm_send(worker_connects[gpu_id], (void *)&task[i], sizeof(task[0])) >
              0 &&
          gmm_recv(worker_connects[gpu_id], (void *)&ret, sizeof(ret)) > 0) {
      }
    }

    // let orig_stream wait on each worker_evt (recorded on each worker-stream)
    if (orig_stream) {
      LOGGER(INFO, "submit user-stream wait on evt:%p", mp_evt[i]);
      CHECK_DRV(__CF("cuStreamWaitEvent")(orig_stream, mp_evt[i],
                                          CU_EVENT_WAIT_DEFAULT));
    }
  }

  ////////////////////////// post handling
  if (orig_stream) {
    // 4.1 optionally, record post-evt on orig stream
    ret = get_evt_at(cur_dev, cur_dev, rsp.data.mp.post_evt_idx, post_evt);
    CHECK_DRV(__CF("cuEventRecord")(post_evt, orig_stream));
    mark_pending_mp(orig_stream);
    LOGGER(DEBUG,
           "post-evt idx:%d cur_dev:%d post_evt:%p stream:%p mark pending done",
           rsp.data.mp.post_evt_idx, cur_dev, post_evt, orig_stream);

  } else {
    // 4.2 for sync, sync mp evt before return
    for (int i = 0; i < task_num; ++i) {
      LOGGER(DEBUG, "sync mode, mp-gpu:%d evt:%p", task[i].worker_dev,
             mp_evt[i]);
      CHECK_DRV(__CF("cuEventSynchronize")(mp_evt[i]));
    }

    LOGGER(DEBUG, "sync mode, sync mp evt done");

    // reclaim resource
    // TODO: mork a special stream '0' at admin to avoid sync other user streams
    // from this same pid
    ret = reclaim_stream(orig_stream, -1);
    auto t2 = std::chrono::system_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    LOGGER(DEBUG, "bytes:%ld dur:%ld BW:%ld MB/sec", bytes, dur.count(),
           bytes / dur.count());
  }

  if (op == GMM_OP_SCATTER) {
    insert_sg_info(rsp.data.mp.gid, cur_dev, tgt_addr, bytes, dm_task,
                   task_num);
  }

  if (op == GMM_OP_GATHER && !orig_stream) {
    delete_sg_info(src_addr);
    // TODO: delete sg when async gather is sync/completed
  }

  return ret;
}

inline void gmm_client_ctx::mark_pending_mp(const CUstream &stream) {
  std::lock_guard<std::mutex> lock_(stream_map_lock);
  stream_map.insert(std::make_pair(stream, true));
}

inline void gmm_client_ctx::reset_pending_mp(const CUstream &stream) {
  std::lock_guard<std::mutex> lock_(stream_map_lock);
  stream_map.insert(std::make_pair(stream, false));
}

inline int gmm_client_ctx::reclaim_evt(int dev_id, gmm_ipc_admin_req &req) {
  int ret = 0;

  req.op = GMM_OP_RECLAIM_EVT;

  // if (gmm_send(admin_connect, (void *)&req, sizeof(req)) > 0) {
  if (gmm_send(admin_connect, (void *)&req, sizeof(req)) > 0 &&
      gmm_recv(admin_connect, (void *)&ret, sizeof(ret)) > 0) {
  }

  return ret;
}

// notify admin to reclaim granted resources (evts) for given client.stream
// (could be null)
// - do sync all pending streams if input client.stream is nullptr
// - could be async, no need to wait for admin done
// TODO: add sync point to avoid actual sync if no new evt added
int gmm_client_ctx::reclaim_stream(const CUstream &user_stream,
                                   int dev_id = -1) {
  int ret = 0;
  if (user_stream && !has_pending_mp(user_stream)) {
    return ret;
  }

  int cur_dev = 0;
  if (dev_id == -1) {
    CHECK_CUDA(cudaGetDevice(&cur_dev));
  }

  gmm_ipc_admin_req req;
  req.data.sync_req.dev_id = (dev_id == -1) ? cur_dev : dev_id;
  req.data.sync_req.pid = pid;
  req.data.sync_req.stream = user_stream;

  if (user_stream) {
    req.op = GMM_OP_SYNC_STREAM;
  } else {
    req.op = GMM_OP_SYNC_DEV;
  }

  // if (gmm_send(admin_connect, (void *)&req, sizeof(req)) > 0
  // if (gmm_send(admin_connect, (void *)&req, sizeof(req)) > 0
  if (gmm_send(admin_connect, (void *)&req, sizeof(req)) > 0) {
    //&& gmm_recv(admin_connect, (void *)&ret, sizeof(ret)) > 0) { //async
  }

  if (user_stream) {
    reset_pending_mp(user_stream);
  }

  return ret;
}

// TODO: to eliminate overhead to query H's baseAddr when
// cuPointerGetAttribute(&base, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
// (CUdeviceptr)host_shm_offset), a possible option:
// pre-malloc big enough virtual mem, e.g. 80GB, very cheap as long as don't
// modify the mem record its baseAddr, notify peer to map when cuMemHostAlloc,
// alloc from above virtual mem, register thus pinned, notify peer to pin when
// H2D, check it's hostAlloc and offset can be quickly - baseAddr note: if ptr
// is malloc w/o register, cuPointerGetAttribute would return
// CUDA_ERROR_INVALID_VALUE(1)
int gmm_client_ctx::register_shm(int cur_dev, gmm_shmInfo_t *&shm,
                                 bool dev_mem) {
  int ret = 0;
  int task_num = 0;

  if (is_sameProcess()) {
    LOGGER(DEBUG, "same process, skip register");
    return ret;
  }

  gmm_ipc_worker_req dm_task[MAX_PATH_NR];
  gmm_ipc_worker_rsp worker_rsp[MAX_PATH_NR];

  gmm_ipc_admin_req req;
  gmm_ipc_admin_rsp rsp;

  // 1.query admin to get proper neighbor
  req.op = GMM_OP_PATH_QUERY_SHM;
  req.data.register_shm_req.pid = pid;
  req.data.register_shm_req.dev_id = cur_dev;

  if (gmm_send(admin_connect, (void *)&req, sizeof(req)) > 0 &&
      gmm_recv(admin_connect, (void *)&rsp, sizeof(rsp)) > 0) {
    task_num = rsp.data.neighbors.path_nr;
    for (int i = 0; i < task_num; ++i) {
      shm->shm_peer[i].dev_id = rsp.data.neighbors.path[i];

      dm_task[i].gmm_op =
          dev_mem ? GMM_OP_REGISTER_SHM_DEV : GMM_OP_REGISTER_SHM_HOST;
      dm_task[i].pid = pid;
      dm_task[i].src_dev = cur_dev;
      dm_task[i].worker_dev = rsp.data.neighbors.path[i];
      dm_task[i].byte = shm->get_size();
      dm_task[i].src_addr = (char *)shm->get_addr();

      if (dev_mem) {
        dm_task[i].shared_fd = shm->get_shmFd();
      } else {
        dm_task[i].shm_idx = shm->get_idx();
      }
    }
    shm->shm_num = task_num;

  } else {
    LOGGER(WARN,
           "pid:%d cur_dev:%d Register shm and query admin return error, "
           "fallback to normal. TODO<---",
           pid, cur_dev);
    return 1;
  }

  gmm_ipc_worker_req *task = &dm_task[0];
  for (int i = 0; i < task_num; ++i) {
    int gpu_id = task[i].worker_dev;
    // 3. submit to workers
    if (dev_mem == false) {  // host
      LOGGER(INFO,
             "pid:%d cur_dev:%d register %s shm to worker:%d op:%d pid:%d "
             "shm_idx:%d",
             pid, cur_dev, dev_mem ? "dev" : "host", gpu_id, task[i].gmm_op,
             task[i].pid, task[i].shm_idx);
      if (gmm_send(worker_connects[gpu_id], (void *)&task[i], sizeof(task[0])) >
              0 &&
          gmm_recv(worker_connects[gpu_id], (void *)&worker_rsp[i],
                   sizeof(worker_rsp[0])) > 0) {
        shm->shm_peer[i].shmInfo_addr = worker_rsp[i].data.shmInfo_addr;
        LOGGER(INFO, "pid:%d cur_dev:%d register host shm to worker:%d done",
               pid, cur_dev, gpu_id);
      } else {
        LOGGER(WARN,
               "pid:%d cur_dev:%d register host shm to worker:%d return error, "
               "continue",
               pid, cur_dev, gpu_id);
      }
    } else {
      // if (gmm_send_fd(worker_connects[gpu_id], cur_dev, task[i]) > 0 &&
      if (gmm_send(worker_connects[gpu_id], (void *)&task[i], sizeof(task[0]),
                   true, task[i].shared_fd) > 0 &&
          gmm_recv(worker_connects[gpu_id], (void *)&worker_rsp[i],
                   sizeof(worker_rsp[0])) > 0) {
        shm->shm_peer[i].shmInfo_addr = worker_rsp[i].data.shmInfo_addr;
        LOGGER(INFO, "pid:%d cur_dev:%d register dev shm to worker:%d done",
               pid, cur_dev, gpu_id);
      } else {
        LOGGER(WARN, "Register dev shm to worker:%d return error, continue",
               gpu_id);
      }
    }
  }

  return ret;
}

int gmm_client_ctx::deregister_shm(gmm_shmInfo_t *&shm, bool dev_mem) {
  int ret = 0;

  if (is_sameProcess()) {
    return ret;
  }

  gmm_ipc_worker_req dm_task[MAX_PATH_NR];
  for (int i = 0; i < shm->shm_num; ++i) {
    dm_task[i].worker_dev = shm->get_peerShm(i).dev_id;

    dm_task[i].gmm_op =
        dev_mem ? GMM_OP_DEREGISTER_SHM_DEV : GMM_OP_DEREGISTER_SHM_HOST;
    dm_task[i].shmInfo_addr_src = shm->get_peerShm(i).shmInfo_addr;
    dm_task[i].pid = pid;
  }

  gmm_ipc_worker_req *task = &dm_task[0];
  for (int i = 0; i < shm->shm_num; ++i) {
    int gpu_id = task[i].worker_dev;
    // 3. submit to worker threads
    if (gmm_send(worker_connects[gpu_id], (void *)&task[i], sizeof(task[0])) >
        0) {
      // if (gmm_recv(worker_connects[gpu_id], (void *)&ret, sizeof(ret)) > 0)
    }
  }

  return ret;
}

// src_addr is the key to find cached data.
// tgt_addr is the address to fetch to. Usually, tgt_addr should be equal to
// src_addr, as we want to fetch the cached data to the original buffer.
int gmm_client_ctx::fetch(char *tgt_addr, char *src_addr, size_t bytes,
                          CUstream &stream) {
  int dev_idx = 0;
  CHECK_DRV(cuPointerGetAttribute(&dev_idx, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                  (CUdeviceptr)tgt_addr));
  // TODO: adapt to src_addr.
  return mp_dma_internal(tgt_addr, dev_idx, src_addr, ANY_DEV, bytes, stream,
                         GMM_OP_GATHER);
}

int gmm_client_ctx::evict(char *src_addr, size_t bytes, CUstream &stream) {
  int dev_idx = 0;
  CHECK_DRV(cuPointerGetAttribute(&dev_idx, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                  (CUdeviceptr)src_addr));
  return mp_dma_internal(nullptr, dev_idx, src_addr, ANY_DEV, bytes, stream,
                         GMM_OP_SCATTER);
}

int gmm_client_ctx::htod_async(char *tgt_addr, char *host_addr, size_t bytes,
                               CUstream &stream) {
  if (!mp_ok(bytes)) {
    return 1;
  }

  CUDA_devMem *dev_ent = find_devMemEntry((CUdeviceptr)tgt_addr);
  if (dev_ent == nullptr) {
    printf("--%s find_devMemEntry fail, ptr:%p\n", __func__, tgt_addr);
  }
  host_mem *host_ent = find_hostMemEntry(host_addr);
  if (dev_ent && host_ent && gdr_ok(dev_ent, host_ent, bytes)) {
    return gmm_gdr_htod(dev_ent->get_mHandle_ref(), dev_ent->get_va_dptr_ref(),
                        host_addr, bytes);
  } else if (dev_ent && host_ent && mp_ok(bytes)) {
    int dev_idx;
    CHECK_DRV(cuPointerGetAttribute(
        &dev_idx, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (CUdeviceptr)tgt_addr));
    int ret = 0;
    ret = mp_dma_internal(tgt_addr, dev_idx, host_addr, CPU_DEV, bytes, stream,
                          GMM_OP_H2D);
    return ret;
  }

  // mp_ok(bytes));
  return 1;
}

int gmm_client_ctx::dtoh_async(char *host_addr, char *src_addr, size_t bytes,
                               CUstream &stream) {
  int dev_idx;
  CHECK_DRV(cuPointerGetAttribute(&dev_idx, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                  (CUdeviceptr)src_addr));
  return mp_dma_internal(host_addr, CPU_DEV, src_addr, dev_idx, bytes, stream,
                         GMM_OP_D2H);
}

// check whether need to goto MP
// -1: not necessary, as high BW exist
// >=0: needed, and return the dev ID acting as inter-dev btw src and tgt
int gmm_client_ctx::dtod_mp_ok(char *tgt_addr, char *src_addr) {
  int ret = -1;
  int src_dev, tgt_dev;
  CHECK_DRV(cuPointerGetAttribute(&src_dev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                  (CUdeviceptr)src_addr));
  CHECK_DRV(cuPointerGetAttribute(&tgt_dev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                  (CUdeviceptr)tgt_addr));

  return (config_shm->DD_MP[src_dev][tgt_dev] == DD_PATH_P2P)
             ? ret
             : config_shm->DD_MP[src_dev][tgt_dev];
  /*
  if (config_shm->DD_MP[src_dev][tgt_dev] != DD_PATH_P2P) {
    ret = config_shm->DD_MP[src_dev][tgt_dev];
  }
  return ret;
  */
}

int gmm_client_ctx::dtod_async(char *tgt_addr, char *src_addr, size_t bytes,
                               CUstream &stream) {
  // if (!mp_ok(bytes)) return 1;

  int src_dev, tgt_dev;
  CHECK_DRV(cuPointerGetAttribute(&src_dev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                  (CUdeviceptr)src_addr));
  CHECK_DRV(cuPointerGetAttribute(&tgt_dev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                  (CUdeviceptr)tgt_addr));
  return mp_dma_internal(tgt_addr, tgt_dev, src_addr, src_dev, bytes, stream,
                         GMM_OP_D2D);
}

// perform H2DA via multi-path
int gmm_client_ctx::htod_async(char *tgt_addr, int tgt_d, char *host_addr,
                               size_t bytes, gmm_req_t *&req_out) {
  return mp_internal(tgt_addr, tgt_d, host_addr, CPU_DEV, bytes, req_out,
                     GMM_OP_H2D);
}

// perform D2HA via multi-path
int gmm_client_ctx::dtoh_async(char *host_addr, char *src_addr, int src_d,
                               size_t bytes, gmm_req_t *&req_out) {
  return mp_internal(host_addr, CPU_DEV, src_addr, src_d, bytes, req_out,
                     GMM_OP_D2H);
}

// perform D2DA via multi-path
int gmm_client_ctx::dtod_async(char *tgt_addr, int tgt_d, char *src_addr,
                               int src_d, size_t bytes, gmm_req_t *&req_out) {
  return mp_internal(tgt_addr, tgt_d, src_addr, src_d, bytes, req_out,
                     GMM_OP_D2D);
}

int gmm_client_ctx::scatter(char *host_addr, char *dev_addr, size_t bytes,
                            const CUstream &stream) {
  int dev_idx = 0;
  CHECK_DRV(cuPointerGetAttribute(&dev_idx, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                  (CUdeviceptr)dev_addr));
  return mp_dma_internal(host_addr, dev_idx, dev_addr, ANY_DEV, bytes, stream,
                         GMM_OP_SCATTER);
}

int gmm_client_ctx::gather(char *dev_addr, char *host_addr, size_t bytes,
                           const CUstream &stream) {
  int dev_idx = 0;
  CHECK_DRV(cuPointerGetAttribute(&dev_idx, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                  (CUdeviceptr)dev_addr));
  return mp_dma_internal(dev_addr, dev_idx, host_addr, ANY_DEV, bytes, stream,
                         GMM_OP_GATHER);
}

// offload given data at src_addr to neighbor via multi-path
int gmm_client_ctx::scatter(char *src_addr, int src_d, size_t bytes,
                            gmm_req_t *&req_out) {
  // set tgt_addr as nullptr, later gmm_split_tasks actually get the offset,
  // finally worker picks up the final base address
  return mp_internal(nullptr, ANY_DEV, src_addr, src_d, bytes, req_out,
                     GMM_OP_SCATTER);
}

// prefetch req.data to given location at tgt_addr via multi-path
// optionally, delete req
int gmm_client_ctx::gather(char *tgt_addr, int tgt_d, size_t bytes,
                           gmm_req_t *&req) {
  return mp_internal(tgt_addr, tgt_d, nullptr, ANY_DEV, bytes, req,
                     GMM_OP_GATHER);
}

// blocking current thread until req_in done
int gmm_client_ctx::synchronize(int cur_dev, gmm_req_t *req) {
  int ret = 0;

  ASSERT(req, "Failed on invalid req");

  for (int i = 0; i < req->task_num; i++) {
    gmm_ipc_worker_req *p = &req->dm_task[i];

    if (p->cross_process) {
      CHECK_DRV(
          __CF("cuEventSynchronize")(mp_evt[p->worker_dev][p->worker_evt_idx]));
    } else {
      // LOGGER(INFO, "intra,%d to sync evt_id:%d", i, p->worker_evt_idx);
      CHECK_DRV(__CF("cuEventSynchronize")(
          config_shm->gmm_evt[p->worker_dev].evt[p->worker_evt_idx]));
      /*
      LOGGER(INFO, "sync req:%ld task:%d num:%d dev:%d evt_id:%d
      cross-process:%d evt:%p", req->req_id, i, req->task_num, p->dev,
      p->evt_id, p->cross_process, &config_shm->evt[p->dev].evt[p->evt_id]);
       */
    }
  }
  // LOGGER(INFO, "sync done");
  return ret;
}

// let stream wait on data moving for req
int gmm_client_ctx::streamWait(int cur_dev, const cudaStream_t &stream,
                               gmm_req_t *req) {
  int ret = 0;
  for (int i = 0; i < req->task_num; i++) {
    gmm_ipc_worker_req *p = &req->dm_task[i];
    // LOGGER(INFO, "streamWait req:%ld task:%d num:%d dev:%d evt:%d",
    // req->req_id, i, req->task_num, p->worker_dev, p->worker_evt_idx);
    if (p->cross_process) {
      CHECK_DRV(__CF("cuStreamWaitEvent")(
          stream, mp_evt[p->worker_dev][p->worker_evt_idx]));
    } else {
      CHECK_DRV(__CF("cuStreamWaitEvent")(
          stream, config_shm->gmm_evt[p->worker_dev].evt[p->worker_evt_idx]));
    }
  }
  return ret;
}

// query req status
int gmm_client_ctx::query(int cur_dev, const gmm_req_t *&req) {
  int ret = 0;
  for (int i = 0; i < req->task_num; i++) {
    const gmm_ipc_worker_req *p = &req->dm_task[i];
    // LOGGER(INFO, "query req:%ld task:%d num:%d dev:%d evt:%d", req->req_id,
    // i, req->task_num, p->worker_dev, p->worker_evt_idx);
    if (p->cross_process) {
      if (cudaEventQuery(mp_evt[p->worker_dev][p->worker_evt_idx]) !=
          cudaSuccess) {
        ret = -1;
        break;
      }
    } else {
      if (cudaEventQuery(
              config_shm->gmm_evt[p->worker_dev].evt[p->worker_evt_idx]) !=
          cudaSuccess) {
        ret = -1;
        break;
      }
    }
  }

  return ret;
}

// entry for pinned host mem alloc
int gmm_client_ctx::hostMem_alloc(CUdevice cur_dev, void **&pp, size_t bytesize,
                                  unsigned int Flags) {
  int ret = 0;
  CUresult rst = CUDA_SUCCESS;
  host_mem_type type = HOST_MEM_TYPE_INVALID;

  if (mp_ok(bytesize)) {
    // 1. alloc host mem via mmap(file),
    // 2. register so to pin
    // 3. notify nvlink peer workers to register shm: mmap and also register/pin
    // 4. record into a shmTable that mapbaseAdd:addrInfo(fd, sz, ), and a set
    // about (baseAlloc, length)
    // 5. when H2D/D2H:
    //  if sz>=32MB, and H is in shm range, notify MP peer about the map_offset
    //  workers: get its local map_baseAddr, then + pidmap_offset
    char shm_name[MAX_SHM_PATH_LEN];
    gmm_shmInfo_t *shm = new gmm_shmInfo_t(GMM_IPC_MEM_HOST_PIN, cur_dev,
                                           get_pid(), get_shmIdx(), bytesize);
    snprintf(shm_name, MAX_SHM_PATH_LEN - 1, "%s_%d_%d", GMM_HOST_SHM_PREFIX,
             shm->get_pid(), shm->get_idx());

    if (0 == gmm_shmCreate(shm_name, bytesize, shm)) {
      rst = __CF("cuMemHostRegister")(
          shm->get_addr(), bytesize,
          CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP);
      if (rst == CUDA_SUCCESS && register_shm(cur_dev, shm, false) == 0) {
        add_shmEntry(shm);
        *pp = shm->get_addr();
        LOGI(
            "pid:%d dev:%d HostAlloc:%p bytes:%ld flag:%d shm_idx:%d "
            "ctx.idx:%d",
            getpid(), cur_dev, *pp, bytesize, Flags, shm->get_idx(),
            get_shmIdx());
        type = HOST_MEM_TYPE_PINNED_SHM;
        goto out;
      } else {
        LOGW(
            "pid:%d dev:%d failed to alloc host pinned shm byte:%ld addr:%p "
            "rst:%d",
            get_pid(), cur_dev, bytesize, shm->get_addr(), rst);
        gmm_shmClose(shm);
      }
    } else {
      LOGI("pid:%d dev:%d host shm create failed, bytes:%ld", pid, cur_dev,
           bytesize);
    }
    delete shm;
  }

  rst = __CF("cuMemHostAlloc")(
      pp, bytesize, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP);
  if (rst == CUDA_SUCCESS) {
    type = HOST_MEM_TYPE_PINNED;
  } else {
    return (int)rst;
  }

out:
  host_mem *ent = new host_mem(CPU_DEV, *pp, bytesize, bytesize, type);
  if (ent) {
    add_hostMemEntry(ent);
  }
  return ret;
}

// entry for pinned host mem free
int gmm_client_ctx::hostMem_free(void *&p) {
  // 1. notify peer workers to de-register
  // 2. deregister, unpin, unmap/free
  // 3. remove record
  CUresult rst = CUDA_SUCCESS;
  gmm_shmInfo_t *shm = find_shmEntry(p);
  if (shm) {
    int ret = deregister_shm(shm, false);  // async?
    rst = __CF("cuMemHostUnregister")(shm->get_addr());
    gmm_shmClose(shm);
    del_shmEntry(p);
    del_hostMemEntry(p);
    shm = nullptr;
    return 0;
  } else {
    rst = __CF("cuMemFreeHost")(p);

    host_mem *ent = find_hostMemEntry(p);
    if (ent) {
      del_hostMemEntry(p);
    }
    return 0;
  }
}

// perform any pre-check before allocation such as quota-check
// ret 0: success
//    >0: failure
int gmm_client_ctx::exec_devMem_preAlloc(CUdevice cur_dev, size_t bytesize) {
  int ret = 0;
  return ret;
}

// exec dev mem allocation
// ret 0: success
//    >0: failure
CUresult gmm_client_ctx::exec_devMem_alloc(CUdevice cur_dev, size_t bytesize,
                                           CUDA_devMem *&ent) {
  cuda_mem_type type = GMM_MEM_TYPE_DEFAULT;
  CUmemGenericAllocationHandle vmm_handle;
  CUdeviceptr dptr;
  size_t alloc_size;

  bool enable_gdr = (gdr_handle && bytesize <= client_cfg->get_GDR_max_sz());
  CUresult rst = gmm_cuda_vmm_alloc(bytesize, cur_dev, dptr, vmm_handle,
                                    &alloc_size, enable_gdr);
  if (rst == CUDA_SUCCESS) {
    client_cfg->inc_alloc_size(cur_dev, alloc_size);
  } else {
    // We return error code to client when GPU memory is used up, which has the
    // same behavior as cudaMalloc. Maybe we can use the following methods
    // (e.g., CUDA unified memory) to handle this case in the future.
    printf("WARN: %s() alloc GPU memory fail. size=%zu, dev=%d\n", __func__,
           bytesize, cur_dev);
    return rst;

    LOGI(
        "pid:%d GPU dev:%d alloc size:%ld failed, CUDA error:%d try other mem "
        "type",
        pid, cur_dev, alloc_size, rst);
    alloc_size = bytesize;

    if (client_cfg->get_OOM_HOSTALLOC()) {
      rst = __CF("cuMemHostAlloc")(
          &dptr, bytesize,
          CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP);
      type = GMM_MEM_TYPE_ZEROCOPY;
    } else {
      rst = __CF("cuMemAllocManaged")(dptr, bytesize, CU_MEM_ATTACH_GLOBAL);
      type = GMM_MEM_TYPE_UM;
    }

    if (rst != CUDA_SUCCESS) {
      LOGI("pid:%d GPU dev:%d alloc size:%ld type:%d failed, CUDA error:%d",
           pid, cur_dev, alloc_size, type, rst);
    }
  }

  if (rst == CUDA_SUCCESS) {
    ent =
        new CUDA_devMem(cur_dev, dptr, vmm_handle, bytesize, alloc_size, type);
    add_devMemEntry(ent);
    LOGI("GPU Mem alloc size:%ld dptr:0x%llx type:%d\n", bytesize, dptr, type);
  }

  return rst;
}

// for large alloc
int gmm_client_ctx::devMem_postAlloc_export(CUdevice cur_dev, CUDA_devMem *&ent,
                                            int &shared_fd) {
  CUresult rst = CUDA_SUCCESS;
  CUmemAccessDesc accessDesc;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = cur_dev;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  // TODO:may only accessable to NVLink connected dev
  for (int i = 0; i < get_devCnt(); ++i) {
    prop.location.id = i;
    accessDesc.location = prop.location;
    rst = __CF("cuMemSetAccess")(ent->get_addr(), ent->get_alloc_size(),
                                 &accessDesc, 1);
  }

  rst = __CF("cuMemExportToShareableHandle")(
      (void *)&shared_fd, ent->get_vmm_handle(),
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  return (int)rst;
  // TODO: close the fd
}

// exec post action after dev mem allocation succeed
// ret 0: success
//    >0: failure
int gmm_client_ctx::exec_devMem_postAlloc(CUdevice cur_dev, CUDA_devMem *&ent) {
  // every successful alloc has an CUDA_devMem entry for magt purpose
  // depends on alloc type, size, optimizaiton flag, some of them may have
  // addition post alloc small alloc (<=64KB) : map by GDR large alloc (>= 8MB)
  // : register for IPC if multi-gpu exist medium alloc         : do-nothing
  // optimization for small obj
  int ret = 0;
  size_t alloc_size = ent->get_alloc_size();
  size_t orig_size = ent->get_orig_size();
  if (gdr_handle && orig_size <= client_cfg->get_GDR_max_sz() &&
      (ent->get_type() != GMM_MEM_TYPE_UM) &&
      (ent->get_type() != GMM_MEM_TYPE_ZEROCOPY)) {
    size_t unused;
    ret = gmm_gdr_map(gdr_handle, ent->get_addr_ref(), alloc_size,
                      ent->get_mHandle_ref(), ent->get_gdr_info_ref(),
                      ent->get_va_dptr_ref(), ent->get_map_dptr_ref(), unused);

    ent->set_type(GMM_MEM_TYPE_GDR);
    printf("pid:%d dev mem orig-req size:%ld alloc-size:%ld perf GDR map\n",
           pid, orig_size, alloc_size);
  } else if (mp_ok(ent->get_orig_size()) && get_devCnt() > 1) {
    printf("pid:%d dev mem orig-req size:%ld alloc-size:%ld IPC-shm\n", pid,
           orig_size, alloc_size);
    int shared_fd;
    ret = devMem_postAlloc_export(cur_dev, ent, shared_fd);
    gmm_shmInfo_t *shm =
        new gmm_shmInfo_t(GMM_IPC_MEM_NV_DEV, cur_dev, (void *)ent->get_addr(),
                          ent->get_vmm_handle(), alloc_size, shared_fd);
    if (register_shm(cur_dev, shm, true) == 0) {
      add_shmEntry(shm, true);
      ent->set_type(GMM_MEM_TYPE_IPC);
    } else {
      delete shm;
      ret = 1;
    }
  }

  // exec other post action
  return ret;
}

// entry for dev mem allocation
int gmm_client_ctx::devMem_alloc(CUdevice cur_dev, CUdeviceptr *&dptr,
                                 size_t bytesize) {
  int ret = 0;
  CUresult rst = CUDA_SUCCESS;
  CUDA_devMem *ent = nullptr;

  if ((0 == (ret = exec_devMem_preAlloc(cur_dev, bytesize))) &&
      (CUDA_SUCCESS == (rst = exec_devMem_alloc(cur_dev, bytesize, ent)))) {
    *dptr = ent->get_addr();
    ret = 0;

    // allocation is done, exec any post action
    exec_devMem_postAlloc(cur_dev, ent);
    return ret;
  }

  return (rst == CUDA_SUCCESS) ? ret : (int)rst;
}

// pre action before free the dev mem
int gmm_client_ctx::exec_devMem_preFree(CUdevice cur_dev, CUdeviceptr dptr,
                                        CUDA_devMem *&ent) {
  // 1. notify peer workers to de-register
  // 2. deregister, unpin, unmap/free
  // 3. remove record
  gmm_shmInfo_t *shm = find_shmDevEntry(dptr);
  if (shm && deregister_shm(shm, true)) {
    // TODO: any drain
    del_shmDevEntry(dptr);
  }

  // unmap GDR if exist
  ent = find_devMemEntry(dptr);
  if (ent == nullptr) {
    printf("--[%s] ent is null, dptr:%p\n", __func__, dptr);
  }
  if (ent && ent->get_type() == GMM_MEM_TYPE_GDR) {
    gmm_gdr_unmap(gdr_handle, ent->get_mHandle_ref(), ent->get_map_dptr_ref(),
                  ent->get_alloc_size());
    ent->set_type(GMM_MEM_TYPE_DEFAULT);
  }
  return 0;
}

CUresult gmm_client_ctx::exec_devMem_free(CUdevice cur_dev, CUdeviceptr dptr,
                                          CUDA_devMem *&ent) {
  if (ent == nullptr) {
    // TODO:why ent is null?
    return CUDA_SUCCESS;
  }

  CUresult rst = CUDA_SUCCESS;
  rst = __CF("cuMemUnmap")(dptr, ent->get_alloc_size());
  rst = __CF("cuMemAddressFree")(dptr, ent->get_alloc_size());
  rst = __CF("cuMemRelease")(ent->get_vmm_handle());

  if (ent->get_type() >= GMM_MEM_TYPE_ALLOC) {
    client_cfg->dec_alloc_size(cur_dev, ent->get_alloc_size());
  }
  return rst;
}

int gmm_client_ctx::exec_devMem_postFree(CUdevice cur_dev, CUdeviceptr dptr,
                                         CUDA_devMem *&ent) {
  del_shmDevEntry(dptr);
  del_devMemEntry(dptr);
  return 0;
}

// entry for dev mem free
int gmm_client_ctx::devMem_free(CUdevice cur_dev, CUdeviceptr dptr) {
  int ret = 0;
  CUresult rst = CUDA_SUCCESS;
  CUDA_devMem *ent = nullptr;

  // TODO: locking
  if ((0 == (ret = exec_devMem_preFree(cur_dev, dptr, ent))) &&
      (CUDA_SUCCESS == (rst = exec_devMem_free(cur_dev, dptr, ent)))) {
    exec_devMem_postFree(cur_dev, dptr, ent);
    return ret;
  }

  return (rst == CUDA_SUCCESS) ? ret : (int)rst;
}

/*int get_dev_bus_id()
 {
   char pciBusId[20];
   CHECK_DRV(cuDeviceGetPCIBusId(pciBusId, 20, src_dev));
}
*/
