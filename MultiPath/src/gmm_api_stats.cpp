#include "gmm_api_stats.h"

void api_stats::print_stat() {
  int i = 0;
  printf("\n");
  for (; i < api_num; i++) {
    printf("API[%d]:%s cnt:%d\n", i, g_stats[i].name, g_stats[i].cnt);
  }

  for (i = 0; i < thread_num; i++) {
    printf("thr[%d]:%lx cnt:%d\n", i, g_thr[i].thrId, g_stats[i].cnt);
  }
}

/*
void __on_exit(int sig)
{
  LOGW("Signal %d captured, xgpu do cleaning up ...", sig);
  signal(SIGTERM, SIG_IGN);
  signal(SIGINT, SIG_IGN);
  signal(SIGPIPE, SIG_IGN);
  signal(SIGBUS, SIG_IGN);
  signal(SIGSEGV, SIG_IGN);
  signal(SIGFPE, SIG_IGN);
  signal(SIGABRT, SIG_IGN);

  xpdk_cleanup();
  exit(0);
}

static int init_handler()
{
  int ret = -1;
  atexit(xpdk_cleanup);
  signal(SIGTERM, __on_exit);
  signal(SIGINT, __on_exit);
  signal(SIGPIPE, SIG_IGN);
  signal(SIGBUS, __on_exit);
  signal(SIGSEGV, __on_exit);
  signal(SIGFPE, __on_exit);
  signal(SIGABRT, __on_exit);
}

void child_handler(int sig)
{
  //signal(SIGCHLD, SIG_IGN);
  pid_t chpid = wait(NULL);

  printf("Child pid %d ended (signal %d)\n", chpid, sig);
}

void child_catch(int signalNumber)
{
    int w_status;
    pid_t w_pid;
    while ((w_pid = waitpid(-1, &w_status, WNOHANG)) != -1 && w_pid != 0) {
      printf("---catch pid %d,return value %d\n", w_pid, WEXITSTATUS(w_status));
//打印子进程PID和子进程返回值 if (WIFEXITED(w_status)) printf("---catch pid
%d,return value %d\n", w_pid, WEXITSTATUS(w_status));
//打印子进程PID和子进程返回值
    }
}

void destr_fn(void *parm)
{
   printf("Destructor function invoked tid:%d pid:%d ppid:%d\n", gettid(),
getpid(), getppid());
}

void signal_handler(int signo)
{
    sigset_t            mask;
    siginfo_t           info;
    printf("Destructor invoked tid:%d pid:%d ppid:%d sig:%d\n", gettid(),
getpid(), getppid(), signo); while (1) { if (sigwaitinfo(&mask, &info) == -1) {
            perror("sigwaitinfo() failed");
            continue;
        }
        switch (info.si_signo) {
        case SIGCHLD:
            // info.si_pid is pid of terminated process, it is not POSIX
            printf("a child terminated, pid = %d\n", info.si_pid);
            break;
        default:
            // should not get here since we only asked for SIGCHLD
            break;
        }
   }
}
*/
