#include "gmm_client_cfg.h"

#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "gmm_api_stats.h"
#include "gmm_client.h"
#include "gmm_common.h"

void gmm_client_cfg_init(void *&libP, gmm_client_cfg *&cfg) {
  cfg = new gmm_client_cfg();
  if (!libP) {
    // try defined path if set, then centos/ubuntu system path
    int fp = -1;
    const char *prefer_path =
        getenv("CUDA_LIB_PATH") ? getenv("CUDA_LIB_PATH") : NULL;
    const char *search_path[] = {prefer_path, "/usr/lib64/libcuda.so.1",
                                 "/usr/lib/x86_64-linux-gnu/libcuda.so.1"};

    for (int i = 0; i < sizeof(search_path) / sizeof(search_path[0]); i++) {
      // skip if null or basic check path name
      if (search_path[i] == NULL ||
          strstr(search_path[i], "libcuda.so") == NULL) {
        continue;
      }

      fp = open(search_path[i], O_RDONLY);
      if (fp != -1) {
        close(fp);

        const char *path = search_path[i];
        libP = dlopen(path, RTLD_NOW | RTLD_GLOBAL);  // RTLD_GLOBAL);
        if (libP) {
          LOGD(
              "%s loaded at %p cuda:%s GMM_MODE:%u GMM_MP:%d visible_GB:%d "
              "pid:%d ppid:%d",
              MODULE_NAME, libP, path, cfg->get_memMode(), cfg->get_MP(),
              cfg->get_UM_GB(), getpid(), getppid());
          break;
        } else {
          LOGI(
              "Failed to dlopen %s: error:%s pls find /usr -name libcuda.so.1 "
              "then set env CUDA_LIB_PATH to the path of libcuda.so.1",
              path, errno > 0 ? strerror(errno) : "n/a");
        }
      } else {
        LOGD("Failed to open %s, error:%s", search_path[i],
             errno > 0 ? strerror(errno) : "n/a");
      }
    }
  }

  ASSERT(libP,
         "Failed to open libcuda.so.1, pls set env CUDA_LIB_PATH to denote its "
         "full path");
}

void gmm_client_cfg_destroy(void *libP) {
  // if (__atomic_dec_cur(&xgpu_refCnt) == 0) {
  int do_ctx_release = getenv("CTX_RESET") ? atoi(getenv("CTX_RESET")) : 0;
  if (do_ctx_release) {
    // CUresult err = __CF("cuDevicePrimaryCtxReset")(0);
  }

  if (libP) {
    dlclose(libP);
  }

  LOGD("%s exit pid:%d ppid:%d===================\n", MODULE_NAME, getpid(),
       getppid());
}
