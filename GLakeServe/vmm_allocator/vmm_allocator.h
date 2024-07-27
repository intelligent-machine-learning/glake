#pragma once

#include <iostream>
#include <vector>
#include <queue>
#include <set>
#include <pthread.h>
#include <cuda.h>
#include <string>
#include <sys/mman.h>
#include <atomic>
#include <sys/stat.h>
#include <sched.h>
#include <fcntl.h>
#include <unistd.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define LOGE(format, ...) fprintf(stdout, "L%d:" format "\n", __LINE__, ##__VA_ARGS__); fflush(stdout);
#define ASSERT(cond, ...) { if(!(cond)) { LOGE(__VA_ARGS__); assert(0); } }
#define WARN(cond, ...) { if(!(cond)) { LOGE(__VA_ARGS__); } }

#define DRV_CALL(call)                                                                                  \
    {                                                                                                   \
        CUresult result = (call);                                                                      \
        if (CUDA_SUCCESS != result)                                                                    \
        {                                                                                              \
            const char *errMsg; cuGetErrorString(result, &errMsg);                                     \
            ASSERT(0, "Error when exec " #call " %s-%d code:%d err:%s", __FUNCTION__, __LINE__, result, errMsg); \
        }                                                                                              \
    }

#define DRV_CALL_RET(call, status_val)                                                                   \
    {                                                                                                    \
            CUresult result = (call);                                                                    \
            if (CUDA_SUCCESS != result)                                                                  \
            {                                                                                            \
                const char *errMsg; cuGetErrorString(result, &errMsg);                                   \
                WARN(0, "Error when exec " #call " %s-%d code:%d err:%s", __FUNCTION__, __LINE__, result, errMsg); \
            }                                                                                            \
            status_val = result;                                                                         \
    }


static constexpr size_t granularitySize = 2 * 1024 * 1024;
//pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

class SpinLock {
public:
    SpinLock() noexcept;
    void lock() noexcept;
    bool try_lock() noexcept;
    void unlock() noexcept;

private:
    std::atomic<bool> lock_;
};

bool init_physical_handle(int num_blocks, int device_id);

class PhyBlock {
  public:
    PhyBlock(int device_id);

    ~PhyBlock();


    int device_id;
    size_t block_size;
    CUmemGenericAllocationHandle alloc_handle;
    CUresult status;
};

std::vector<std::shared_ptr<PhyBlock>> phy_block_cache;

class KVcacheSegment {
  public:
    KVcacheSegment(int layer_num, size_t context_max_size, size_t expand_size, bool expand);

    bool expandMultiSegment(int expand_num);

    CUdeviceptr getKeyDevicePtr(int layer_index);
    CUdeviceptr getValueDevicePtr(int layer_index);

    void copy_kv_cache(size_t token_size, size_t kv_size);

    void release_seq(int seq_id);
    void preempt_seq(int seq_id);

    uint64_t get_used_slot();
    void set_used_size(size_t used_size);

    void write_string(std::string str);
    void read_string();
	void eagerExtend(size_t token_size, int seq_id);
	std::vector<int> show_info(int seq_id);
    int layer_num;
    size_t context_max_size;
    int device_id;
    size_t actual_size;
    size_t used_size;
   	size_t expand_size; 
    std::unordered_map<int, std::vector<std::shared_ptr<PhyBlock>>> k_layer_phy_blocks;
    std::unordered_map<int, std::vector<std::shared_ptr<PhyBlock>>> v_layer_phy_blocks;
    CUresult status;
    std::vector<CUdeviceptr> k_layer_segment_ptr;
    std::vector<CUdeviceptr> v_layer_segment_ptr;
};


class ExtendRequest {
    public:
    ExtendRequest(KVcacheSegment* kv_cache_segment, int expand_size, int seq_id, int op) :
                                                kv_cache_segment_(kv_cache_segment),
                                                expand_size_(expand_size),
                                                seq_id_(seq_id),
												op_(op) {}

    KVcacheSegment* kv_cache_segment_;
    int expand_size_;
    int seq_id_;
	int op_; // 0: extend 1: preempt 2: release
};

class KVCache {
  public:
    KVCache(int num_seqs);

    void AddKVCache(int slot, CUdeviceptr kv_cache);

    CUdeviceptr getKVCachePtr();

    int num_seqs;
    CUdeviceptr kv_cache_ptr;
};


std::vector<int> extend_kv_cache_batch(std::vector<KVCache*> k_cache_batch, std::vector<KVCache*> v_cache_batch, std::vector<int> slot_mapping, 
                                       std::vector<size_t> token_size, std::vector<KVcacheSegment*> segment, std::vector<int> set_used_size, size_t kv_size, bool gen) {
    int num_layer = k_cache_batch.size();
    int num_seq = slot_mapping.size();
    std::vector<int> slot_m;
    for (int i = 0; i < num_seq; i++) {
        KVcacheSegment* tmp_segment = segment[i];
        if (set_used_size[i] > 0) {
            tmp_segment->set_used_size(set_used_size[i] * kv_size);
        }
        slot_m.push_back(tmp_segment->get_used_slot());
        if (!gen) {
            tmp_segment->copy_kv_cache(token_size[i], kv_size);
        } else {
            tmp_segment->used_size += token_size[i];
        }
        //:tmp_segment->copy_kv_cache(token_size[i]);
    }
    //for (int i = 0; i < num_layer; i++) {
    //    KVCache* k_cache = k_cache_batch[i];
    //    KVCache* v_cache = v_cache_batch[i];
    //    for (int j = 0; j < num_seq; j++) {
    //        KVcacheSegment* tmp_segment = segment[j];
    //        int slot = slot_mapping[j];            
    //        k_cache->AddKVCache(slot, tmp_segment->getKeyDevicePtr(i));
    //        v_cache->AddKVCache(slot, tmp_segment->getValueDevicePtr(i));
    //    }
    //}
    return slot_m;

}

void copy_kv_cache(std::vector<KVCache*> k_cache_batch, std::vector<KVCache*> v_cache_batch, std::vector<KVcacheSegment*> segment) {
    int num_layer = k_cache_batch.size();
    int num_seq = segment.size();
    for (int i = 0; i < num_layer; i++) {
        KVCache* k_cache = k_cache_batch[i];
        KVCache* v_cache = v_cache_batch[i];
        for (int j = 0; j < num_seq; j++) {
            KVcacheSegment* tmp_segment = segment[j];
            k_cache->AddKVCache(j, tmp_segment->getKeyDevicePtr(i));
            v_cache->AddKVCache(j, tmp_segment->getValueDevicePtr(i));
        }
    }
}

std::vector<int> print_shm_state(std::vector<int> seq_ids);
bool query_shm(std::vector<int> seq_ids);
bool query_cycle(std::vector<int> seq_ids);
std::vector<bool> query_unmap(std::vector<int> seq_ids);
int showFreeBlocks();

PYBIND11_MODULE( vmmAllocator, m ){
    m.doc() = "vmmAllocator";

    pybind11::class_<KVCache>(m, "KVCache")
        .def( pybind11::init<int>() )
        .def ("getKVCachePtr", &KVCache::getKVCachePtr );
    
    pybind11::class_<KVcacheSegment>(m, "KVcacheSegment")
        .def( pybind11::init<int, size_t, size_t, bool>() )
        .def( "getKeyDevicePtr", &KVcacheSegment::getKeyDevicePtr )
        .def( "getValueDevicePtr", &KVcacheSegment::getValueDevicePtr )
        .def( "get_used_slot", &KVcacheSegment::get_used_slot )
        .def( "set_used_size", &KVcacheSegment::set_used_size )
        .def( "write_string", &KVcacheSegment::write_string )
        .def( "read_string", &KVcacheSegment::read_string )
        .def( "preempt_seq", &KVcacheSegment::preempt_seq )
	    .def( "release_seq", &KVcacheSegment::release_seq )
		.def( "eagerExtend", &KVcacheSegment::eagerExtend )
		.def( "show_info", &KVcacheSegment::show_info );


    m.def("extend_kv_cache_batch", &extend_kv_cache_batch, "extend_kv_cache_batch");
    m.def("init_physical_handle", &init_physical_handle, "init Physical Handle");
    m.def("copy_kv_cache", &copy_kv_cache, "copy_kv_cache");
	m.def("query_shm", &query_shm, "query_shm");
	m.def("query_cycle", &query_cycle, "query_cycle");
	m.def("query_unmap", &query_unmap, "query_unmap");
	m.def("print_shm_state", &print_shm_state, "print_shm_state");
	m.def("showFreeBlocks", &showFreeBlocks, "showFreeBlocks");
}
