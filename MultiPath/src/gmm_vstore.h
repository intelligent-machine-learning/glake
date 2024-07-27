// for tiering across multi-dev
#pragma once

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "gmm_common.h"
#include "gmm_queue.h"

// composite addr as key : client_slot(high 16b) | addr( low 48bit)
typedef uint64_t comb_addr;
#define X86_VALID_VA_BITS (48)
#define X86_VALID_VA_MASK (~(0x1ULL << X86_VALID_VA_BITS))

#define get_comb_addr(slot, addr) \
  ((addr & X86_VALID_VA_MASK) | (((uint64_t)slot) << X86_VALID_VA_BITS))

enum gmm_vstore_state {
  VSTORE_INIT = 0,

  VSTORE_ASSIGN_DEV = 1,   /* assign by admin*/
  VSTORE_WORKER_SAVED = 2, /* saved by worker*/

  VSTORE_SWAPPING_OUT, /*swapping out by worker*/
  VSTORE_SWAPPED_OUT,
  VSTORE_SWAPPING_IN, /*swapping in by worker*/

  VSTORE_INVALID,
};

// info to describe a tiering range at a worker
// (i.e., split from a store req)
struct gmm_vstore_entry {
  gmm_id gid;  // unique ID

  comb_addr c_hAddr;  // client_slot(16b) | evental host addr(48bit)

  uint64_t cur_addr;  // current range store addr, CPU/GPU/...
  int worker_id;  // current location, -1: CPU; or neighbor dev_id (worker_id)
  int orig_dev;   // dev id where data come from

  // set by client when split as ranges
  uint64_t range_size;
  uint32_t base_offset;
  uint32_t range_split_offset;

  char tot_range : 8;
  gmm_vstore_state state;
  // others: timestamp, ...

 public:
  gmm_id get_storeID() const { return gid; }

  int get_workerID() const { return worker_id; }
  int get_origDev() const { return orig_dev; }
  char get_rangeNum() const { return tot_range; }
  gmm_vstore_state get_state() const { return state; }

  uint64_t get_curAddr() const { return cur_addr; }
  uint64_t get_rangeSize() const { return range_size; }
  uint64_t get_baseOffset() const { return base_offset; }
  uint64_t get_splitOffset() const { return range_split_offset; }

  uint16_t get_slot_idx() const {
    return (uint16_t)(c_hAddr >> X86_VALID_VA_BITS);
  }
  uint64_t get_hostAddr() const { return (c_hAddr & X86_VALID_VA_MASK); }
  comb_addr get_c_hAddr() const { return c_hAddr; }

  void set_workerID(int id_) { worker_id = id_; }
  void set_rangeNum(int cnt) { tot_range = cnt; }
  void set_rangeSize(uint64_t size_) { range_size = size_; }
  void set_baseOffset(uint64_t offset_) { base_offset = offset_; }
  void set_splitOffset(uint64_t offset_) { range_split_offset = offset_; }
  void set_state(gmm_vstore_state state_) { state = state_; }

  void print() {
    printf("ID:%lx C_Addr:%lx dev:%d range_cnt:%d\n", gid, c_hAddr, orig_dev,
           tot_range);
  }

 public:
  gmm_vstore_entry() { printf("vstore default construct\n"); }
  gmm_vstore_entry(const gmm_vstore_entry &a) {
    printf("vstore copy construct\n");
  }

  // created by admin
  gmm_vstore_entry(uint64_t id, comb_addr hAddr, int orig_dev_id, int cnt) {
    // printf("new entry id:%lx addr:%lx dev:%d cnt:%d\n", id, hAddr,
    // orig_dev_id, cnt);
    gid = id;
    c_hAddr = hAddr;
    orig_dev = orig_dev_id;
    tot_range = (char)cnt;

    cur_addr = 0ULL;
    worker_id = -1;

    range_size = 0UL;
    range_split_offset = 0UL;
    base_offset = 0UL;
    state = VSTORE_INIT;
  }

  // created by client/worker
  gmm_vstore_entry(gmm_id id, comb_addr hAddr, int orig_dev_id, int range_cnt,
                   int worker_dev, size_t split_offset, size_t baseAlloc_offset,
                   size_t range_sz, size_t cur_addr_,
                   gmm_vstore_state init_state) {
    // printf("new entry id:%lx addr:%lx dev:%d cnt:%d\n", id, hAddr,
    // orig_dev_id, range_cnt);
    gid = id;
    c_hAddr = hAddr;
    orig_dev = orig_dev_id;
    tot_range = (char)range_cnt;

    cur_addr = 0ULL;
    worker_id = worker_dev;
    range_size = range_sz;

    range_split_offset = split_offset;
    base_offset = baseAlloc_offset;
    cur_addr = cur_addr_;
    state = init_state;
  }
};

// a K:V map to manage temp store entries, e.g. CRID
class gmm_vstore_mgr {
 public:
  std::atomic<uint64_t> store_num;  // always inc for a new store req

  size_t tot_store_bytes;
  size_t tot_load_bytes;
  // TODO: stats for each worker

  std::mutex lock;
  // each is a vector (range []), key by comb_addr
  std::unordered_map<comb_addr, std::vector<gmm_vstore_entry *> *> vstores;

 public:
  gmm_vstore_mgr() {
    store_num = 0ULL;
    tot_store_bytes = 0;
    tot_load_bytes = 0;
  }

  // created by admin (optional, for debug)
  std::vector<gmm_vstore_entry *> *new_store(uint16_t c_slot_idx, int c_dev,
                                             uint64_t c_hostAddr,
                                             uint64_t bytes, int *workers,
                                             int worker_cnt) {
    std::vector<gmm_vstore_entry *> *stores =
        new std::vector<gmm_vstore_entry *>();

    comb_addr c_addr = get_comb_addr(c_slot_idx, c_hostAddr);
    stores->reserve(worker_cnt);

    for (int i = 0; i < worker_cnt; ++i) {
      gmm_vstore_entry *ent =
          new gmm_vstore_entry(store_num++, c_addr, c_dev, worker_cnt);
      ent->set_workerID(workers[i]);
      stores->push_back(ent);
    }
    // printf("create ents done cnt:%d %ld\n", worker_cnt, stores->size());

    insert_store(stores);
    tot_store_bytes += bytes;
    return stores;
  }

  // by client
  // pre: worker assigned
  std::vector<gmm_vstore_entry *> *new_store(gmm_id gid, uint16_t c_slot_idx,
                                             int c_dev, uint64_t c_hostAddr,
                                             uint64_t bytes,
                                             struct gmm_ipc_worker_req *workers,
                                             int worker_cnt) {
    std::vector<gmm_vstore_entry *> *stores =
        new std::vector<gmm_vstore_entry *>();

    comb_addr c_addr = get_comb_addr(c_slot_idx, c_hostAddr);
    stores->reserve(worker_cnt);

    for (int i = 0; i < worker_cnt; ++i) {
      gmm_vstore_entry *ent = new gmm_vstore_entry(
          gid, c_addr, c_dev, worker_cnt, workers[i].worker_dev,
          workers[i].split_offset_src, workers[i].base_offset_src,
          workers[i].byte, 0UL, VSTORE_ASSIGN_DEV);
      stores->push_back(ent);
    }

    insert_store(stores);
    tot_store_bytes += bytes;
    return stores;
  }

  // by worker
  // pre: alloc devMem cur_addr_
  std::vector<gmm_vstore_entry *> *new_store(
      int worker_dev, gmm_id gid, uint16_t c_slot_idx, int c_dev,
      uint64_t c_hostAddr, uint64_t bytes, size_t split_offset,
      size_t base_offset, size_t cur_addr_) {
    std::vector<gmm_vstore_entry *> *stores =
        new std::vector<gmm_vstore_entry *>();
    comb_addr c_addr = get_comb_addr(c_slot_idx, c_hostAddr);
    stores->reserve(1);

    gmm_vstore_entry *ent =
        new gmm_vstore_entry(gid, c_addr, c_dev, 1, worker_dev, split_offset,
                             base_offset, bytes, cur_addr_, VSTORE_ASSIGN_DEV);
    stores->push_back(ent);
    insert_store(stores);
    tot_store_bytes += bytes;
    return stores;
  }

  // insert store
  inline void insert_store(std::vector<gmm_vstore_entry *> *store) {
    vstores.emplace(store->at(0)->get_c_hAddr(), store);
  }

  // find the store
  inline std::vector<gmm_vstore_entry *> *find_store(uint16_t slot,
                                                     uint64_t addr) {
    comb_addr c_addr = get_comb_addr(slot, addr);

    auto ent = vstores.find(c_addr);
    if (ent != vstores.end()) {
      return ent->second;
    } else {
      return nullptr;
    }
  }

  inline void print_store(uint16_t slot, uint64_t addr) {
    std::vector<gmm_vstore_entry *> *store = find_store(slot, addr);
    if (!store) {
      printf("failed to find store for input slot:%d addr:%lx\n", slot, addr);
      return;
    }

    printf(
        "    ID \t orig-dev \t hAddr \t tot_range \t worker \t size \t "
        "cur_addr \t base_offset \t split_offset \t state\n");
    for (int i = 0; i < store->size(); ++i) {
      printf(
          "%d  %lx \t %d \t %lx \t %d \t %d \t %ld \t %lx \t %lx \t %lx \t "
          "%d\n",
          i, store->at(i)->get_storeID(), store->at(i)->get_origDev(),
          store->at(i)->get_hostAddr(), store->at(i)->get_rangeNum(),
          store->at(i)->get_workerID(), store->at(i)->get_rangeSize(),
          store->at(i)->get_curAddr(), store->at(i)->get_baseOffset(),
          store->at(i)->get_splitOffset(), store->at(i)->get_state());
    }
  }

  // traverse the vstore
  void print_store(bool verbose = false) {
    printf("vstore entry cnt:%ld tot_store_bytes:%ld\n", vstores.size(),
           tot_store_bytes);

    if (verbose) {
      for (auto &kv : vstores) {
      }
    }
  }

  // delete the store entry
  void delete_store(uint16_t slot, uint64_t addr) {
    comb_addr c_addr = get_comb_addr(slot, addr);

    auto ent = vstores.find(c_addr);
    if (ent != vstores.end()) {
      delete ent->second;
      vstores.erase(c_addr);
      // TODO: dec bytes
    }
  }
};
