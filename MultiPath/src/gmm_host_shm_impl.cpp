/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cstdlib>
#include <string>

#include "gmm_host_shm.h"

int gmm_shmCreate(const char *name, size_t sz, gmm_shmInfo_t *&info) {
  int status = 0;

  info->shm_fd = shm_open(name, O_RDWR | O_CREAT, 0777);
  if (info->shm_fd < 0) {
    LOGGER(WARN, "pid:%d failed to create shm:%s error:%s", getpid(), name,
           strerror(errno));
    return errno;
  }

  status = ftruncate(info->shm_fd, sz);
  if (status != 0) {
    LOGGER(WARN, "pid:%d failed to truncate shm:%s error:%s", getpid(), name,
           strerror(errno));
    return status;
  }

  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shm_fd, 0);
  if (info->addr == NULL) {
    LOGGER(WARN, "pid:%d failed to mmap shm:%s error:%s", getpid(), name,
           strerror(errno));
    return errno;
  }

  info->size = sz;
  LOGGER(WARN, "pid:%d mmap shm:%s done", getpid(), name);
  return 0;
}

int gmm_shmOpen(const char *name, size_t sz, gmm_shmInfo_t *&info) {
  info->shm_fd = shm_open(name, O_RDWR, 0777);
  if (info->shm_fd < 0) {
    LOGGER(WARN, "pid:%d failed on open shm:%s error:%s", getpid(), name,
           strerror(errno));
    return errno;
  }

  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shm_fd, 0);
  if (info->addr == nullptr) {
    LOGGER(WARN, "pid:%d failed on mmap shm:%s error:%s", getpid(), name,
           strerror(errno));
    return errno;
  }

  info->size = sz;
  return 0;
}

void gmm_shmClose(gmm_shmInfo_t *&info) {
  if (info->addr) {
    munmap(info->addr, info->size);
    info->addr = nullptr;
  }
  if (info->shm_fd) {
    close(info->shm_fd);
    info->shm_fd = 0;
  }
}

int spawnProcess(Process *process, const char *app, char *const *args) {
  *process = fork();
  if (*process == 0) {
    if (0 > execvp(app, args)) {
      return errno;
    }
  } else if (*process < 0) {
    return errno;
  }
  return 0;
}

int waitProcess(Process *process) {
  int status;
  do {
    if (0 > waitpid(*process, &status, 0)) {
      return errno;
    }
  } while (!WIFEXITED(status));
  return WEXITSTATUS(status);
}

int ipcCreateSocket(ipcHandle *&handle, const char *name,
                    const std::vector<Process> &processes) {
  int server_fd;
  struct sockaddr_un servaddr;

  handle = new ipcHandle;
  memset(handle, 0, sizeof(*handle));
  handle->socket = -1;
  handle->socketName = NULL;

  // Creating socket file descriptor
  if ((server_fd = socket(AF_UNIX, SOCK_DGRAM, 0)) == 0) {
    perror("IPC failure: Socket creation failed");
    return -1;
  }

  unlink(name);
  bzero(&servaddr, sizeof(servaddr));
  servaddr.sun_family = AF_UNIX;

  size_t len = strlen(name);
  if (len > (sizeof(servaddr.sun_path) - 1)) {
    perror("IPC failure: Cannot bind provided name to socket. Name too large");
    return -1;
  }

  strncpy(servaddr.sun_path, name, sizeof(servaddr.sun_path));

  if (bind(server_fd, (struct sockaddr *)&servaddr, SUN_LEN(&servaddr)) < 0) {
    perror("IPC failure: Binding socket failed");
    return -1;
  }

  handle->socketName = new char[strlen(name) + 1];
  strcpy(handle->socketName, name);
  handle->socket = server_fd;
  return 0;
}

int ipcOpenSocket(ipcHandle *&handle) {
  int sock = 0;
  struct sockaddr_un cliaddr;

  handle = new ipcHandle;
  memset(handle, 0, sizeof(*handle));

  if ((sock = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
    perror("IPC failure:Socket creation error");
    return -1;
  }

  bzero(&cliaddr, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;
  char temp[10];

  // Create unique name for the socket.
  sprintf(temp, "%u", getpid());

  strcpy(cliaddr.sun_path, temp);
  if (bind(sock, (struct sockaddr *)&cliaddr, sizeof(cliaddr)) < 0) {
    perror("IPC failure: Binding socket failed");
    return -1;
  }

  handle->socket = sock;
  handle->socketName = new char[strlen(temp) + 1];
  strcpy(handle->socketName, temp);

  return 0;
}

int ipcCloseSocket(ipcHandle *handle) {
  if (!handle) {
    return -1;
  }

  if (handle->socketName) {
    unlink(handle->socketName);
    delete[] handle->socketName;
  }
  close(handle->socket);
  delete handle;
  return 0;
}

int ipcRecvShareableHandle(ipcHandle *handle, ShareableHandle *shHandle) {
  struct msghdr msg = {0};
  struct iovec iov[1];
  struct cmsghdr cm;

  // Union to guarantee alignment requirements for control array
  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  struct cmsghdr *cmptr;
  ssize_t n;
  int receivedfd;
  char dummy_buffer[1];
  ssize_t sendResult;

  msg.msg_control = control_un.control;
  msg.msg_controllen = sizeof(control_un.control);

  iov[0].iov_base = (void *)dummy_buffer;
  iov[0].iov_len = sizeof(dummy_buffer);

  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  if ((n = recvmsg(handle->socket, &msg, 0)) <= 0) {
    perror("IPC failure: Receiving data over socket failed");
    return -1;
  }

  if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) &&
      (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
    if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS)) {
      return -1;
    }

    memmove(&receivedfd, CMSG_DATA(cmptr), sizeof(receivedfd));
    *(int *)shHandle = receivedfd;
  } else {
    return -1;
  }

  return 0;
}

int ipcRecvDataFromClient(ipcHandle *serverHandle, void *data, size_t size) {
  ssize_t readResult;
  struct sockaddr_un cliaddr;
  socklen_t len = sizeof(cliaddr);

  readResult = recvfrom(serverHandle->socket, data, size, 0,
                        (struct sockaddr *)&cliaddr, &len);
  if (readResult == -1) {
    perror("IPC failure: Receiving data over socket failed");
    return -1;
  }
  return 0;
}

int ipcSendDataToServer(ipcHandle *handle, const char *serverName,
                        const void *data, size_t size) {
  ssize_t sendResult;
  struct sockaddr_un serveraddr;

  bzero(&serveraddr, sizeof(serveraddr));
  serveraddr.sun_family = AF_UNIX;
  strncpy(serveraddr.sun_path, serverName, sizeof(serveraddr.sun_path) - 1);

  sendResult = sendto(handle->socket, data, size, 0,
                      (struct sockaddr *)&serveraddr, sizeof(serveraddr));
  if (sendResult <= 0) {
    perror("IPC failure: Sending data over socket failed");
  }

  return 0;
}

int ipcSendShareableHandle(ipcHandle *handle,
                           const std::vector<ShareableHandle> &shareableHandles,
                           Process process, int data) {
  struct msghdr msg;
  struct iovec iov[1];

  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  struct cmsghdr *cmptr;
  ssize_t readResult;
  struct sockaddr_un cliaddr;
  socklen_t len = sizeof(cliaddr);

  // Construct client address to send this SHareable handle to
  bzero(&cliaddr, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;
  char temp[10];
  sprintf(temp, "%u", process);
  strcpy(cliaddr.sun_path, temp);
  len = sizeof(cliaddr);

  // Send corresponding shareable handle to the client
  int sendfd = (int)shareableHandles[data];

  msg.msg_control = control_un.control;
  msg.msg_controllen = sizeof(control_un.control);

  cmptr = CMSG_FIRSTHDR(&msg);
  cmptr->cmsg_len = CMSG_LEN(sizeof(int));
  cmptr->cmsg_level = SOL_SOCKET;
  cmptr->cmsg_type = SCM_RIGHTS;

  memmove(CMSG_DATA(cmptr), &sendfd, sizeof(sendfd));

  msg.msg_name = (void *)&cliaddr;
  msg.msg_namelen = sizeof(struct sockaddr_un);

  iov[0].iov_base = (void *)"";
  iov[0].iov_len = 1;
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  ssize_t sendResult = sendmsg(handle->socket, &msg, 0);
  if (sendResult <= 0) {
    perror("IPC failure: Sending data over socket failed");
    return -1;
  }

  return 0;
}

int ipcSendShareableHandles(
    ipcHandle *handle, const std::vector<ShareableHandle> &shareableHandles,
    const std::vector<Process> &processes) {
  // Send all shareable handles to every single process.
  for (int i = 0; i < shareableHandles.size(); i++) {
    for (int j = 0; j < processes.size(); j++) {
      checkIpcErrors(
          ipcSendShareableHandle(handle, shareableHandles, processes[j], i));
    }
  }
  return 0;
}

int ipcRecvShareableHandles(ipcHandle *handle,
                            std::vector<ShareableHandle> &shareableHandles) {
  for (int i = 0; i < shareableHandles.size(); i++) {
    checkIpcErrors(ipcRecvShareableHandle(handle, &shareableHandles[i]));
  }
  return 0;
}

int ipcCloseShareableHandle(ShareableHandle shHandle) {
  return close(shHandle);
}
