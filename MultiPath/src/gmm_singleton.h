#pragma once

#include <netinet/in.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

class SingletonProcess {
 public:
  SingletonProcess(uint16_t port0)
      : socket_fd(-1), rc(1), port(port0), path(nullptr) {}
  SingletonProcess(char *path_) : socket_fd(-1), rc(1), port(0), path(path_) {}

  ~SingletonProcess() {
    if (socket_fd != -1) {
      close(socket_fd);
    }
  }

  bool operator()() {
    if (socket_fd == -1 || rc) {
      if (path) {
        struct sockaddr_un addr;
        addr.sun_family = AF_UNIX;
        strcpy(addr.sun_path, path);

        if ((socket_fd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
          // std::cout << "Failed to create socket" << path << " "
          // <<strerror(errno) <<std::endl;
        } else {
          // unlink(path); // if unlink() in DP, a different process will able
          // to bind on same admin
          if ((rc = bind(socket_fd, (struct sockaddr *)&addr, sizeof(addr))) !=
              0) {
            // std::cout << "Failed to bind " << path << " "<<strerror(errno)
            // <<std::endl;
          } else {
            // std::cout << "pid:" << getpid() << " bind " << path << "
            // done"<<std::endl;
          }
        }
      } else if (port > 0) {
        socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
        struct sockaddr_in name;
        name.sin_family = AF_INET;
        name.sin_port = htons(port);
        name.sin_addr.s_addr = htonl(INADDR_ANY);
        rc = bind(socket_fd, (struct sockaddr *)&name, sizeof(name));
      }
    }
    return (socket_fd != -1 && rc == 0);
  }

  std::string GetLockFileName() { return "port " + std::to_string(port); }

  int GetSocket() { return socket_fd; }

 private:
  int socket_fd;
  int rc;
  uint16_t port;
  char *path;
};
