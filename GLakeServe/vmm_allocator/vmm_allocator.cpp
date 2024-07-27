#include "vmm_allocator.h"

std::set<int> extend_slot;
std::queue<ExtendRequest> extendQueue;
int shm_fd;

void* shm_ptr;
int* int_ptr;
int* cycle_ptr;
int* unmap_ptr;

SpinLock slock;
SpinLock extend_lock;
SpinLock::SpinLock() noexcept : lock_{false} {}

void SpinLock::lock() noexcept {
    for (;;) {
        if (!lock_.exchange(true, std::memory_order_acquire)) {
            return;
        }
        while (lock_.load(std::memory_order_relaxed)) {
            #ifdef _WIN32
            YieldProcessor();
            #else
            sched_yield();
            #endif
        }
    }
}

bool SpinLock::try_lock() noexcept {
    return !lock_.load(std::memory_order_relaxed) &&
           !lock_.exchange(true, std::memory_order_acquire);
}

void SpinLock::unlock() noexcept {
    lock_.store(false, std::memory_order_release);
}

void* eagerly_extend(void* arg) {
    // when call this function, assume that the original allocation is disabled 
    while (true) {
        if (extendQueue.empty()) {
            continue;
        }
        while(!extendQueue.empty()) {
            ExtendRequest er = extendQueue.front();
            extendQueue.pop();
            if (er.op_ == 0) {
				extend_lock.lock();
            	if (extend_slot.find(er.seq_id_) == extend_slot.end()) {
            	    extend_lock.unlock();
            	    continue;
            	}

            	if (!er.kv_cache_segment_->expandMultiSegment(er.expand_size_)) {
            	    // this should not happen
            	    WARN(0, "async-ly & earger-ly expand segment failed %d \n", er.seq_id_);
            	    extend_lock.unlock();
            	    return NULL;
            	}
            	extend_slot.erase(er.seq_id_);
            	extend_lock.unlock();
            	int_ptr[er.seq_id_] = 1;
        	} else {
				printf("unknown op %d \n", er.op_);
			}

		}
    }
    return NULL;
}

std::vector<int> print_shm_state(std::vector<int> seq_ids) {
	std::vector<int> res;

	for (auto& seq_id : seq_ids) {
		res.push_back(cycle_ptr[seq_id]*10+int_ptr[seq_id]);
	}

	return res;
}


bool query_shm(std::vector<int> seq_ids) {
    bool res = true;
    for (auto& seq_id : seq_ids) {
        res &= int_ptr[seq_id];
    }
    return res;
}

// cycle : need to wait #cycle cycles to check data ready
bool query_cycle(std::vector<int> seq_ids) {
    bool res = false;
    slock.lock();
    for (auto& seq_id : seq_ids) {
        if (cycle_ptr[seq_id] != 0) cycle_ptr[seq_id]--;
        res |= (int_ptr[seq_id] == 0 && cycle_ptr[seq_id] == 0);
    }
    slock.unlock();
    return res;
}

std::vector<bool> query_unmap(std::vector<int> seq_ids) {
	std::vector<bool> res;
	for (auto& seq_id : seq_ids) {
    	res.push_back(!unmap_ptr[seq_id]);
	}	
	return res;
}

bool init_physical_handle(int num_blocks, int device_id) {
    shm_fd = shm_open("/my_shm", O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    shm_ptr = mmap(NULL, sizeof(int)*3000, PROT_WRITE, MAP_SHARED, shm_fd, 0);
    
	int_ptr = reinterpret_cast<int*>(shm_ptr);
    cycle_ptr = int_ptr + 1000;
   	unmap_ptr = cycle_ptr + 1000; 
	
	ftruncate(shm_fd, sizeof(int) * 3000);
    	

    
	for (int i = 0; i < 1000; i++) {
        int_ptr[i] = 1;
    }
    for (int i = 0; i < 1000; i++) {
        cycle_ptr[i] = 0;
    }
	for (int i = 0; i < 1000; i++) {
        unmap_ptr[i] = 0;
    }
	
	for (int i = 0; i < num_blocks; i++) {
        std::shared_ptr<PhyBlock> phy_block = std::make_shared<PhyBlock>(device_id);
        if (phy_block->status != CUDA_SUCCESS) {
            WARN(0, "init_physical_handle failed");
            return false;
        }
        phy_block_cache.emplace_back(std::move(phy_block));
    }
	printf("init_phy: %d\n", num_blocks);
	pthread_t tid;
    int rc = pthread_create(&tid, NULL, eagerly_extend, NULL);
    if (rc != 0) {
        fprintf(stderr, "Error creating thread: %d\n", rc);
        return -1;
    }
    return true;
}


PhyBlock::PhyBlock(int device_id):
            device_id(device_id),
            block_size(granularitySize),
            status(CUDA_SUCCESS)
{
    if (device_id == -1) {
        DRV_CALL(cuCtxGetDevice(&device_id));
    }
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;

    status = cuMemCreate(&alloc_handle, block_size, &prop, 0ULL);
}

PhyBlock::~PhyBlock() {
    if (status == CUDA_SUCCESS) {
        status = cuMemRelease(alloc_handle);
    }
}


KVcacheSegment::KVcacheSegment(int layer_num, size_t context_max_size, size_t expand_size, bool expand) {
    this->layer_num = layer_num;
    this->context_max_size = context_max_size;
	
    device_id = -1;

    this->actual_size = 0;
    this->used_size = 0;
	this->expand_size = expand_size;	
    for (int i = 0;i < this->layer_num; i++) {
        CUdeviceptr key_segment_ptr, value_segment_ptr;
        status = cuMemAddressReserve(&key_segment_ptr, this->context_max_size, 0ULL, 0ULL, 0ULL);
        if (status != CUDA_SUCCESS) {
            WARN(0, "KVcacheSegment reserve ptr failed");
            return;
        }
        this->k_layer_segment_ptr.emplace_back(key_segment_ptr);

        status = cuMemAddressReserve(&value_segment_ptr, this->context_max_size, 0ULL, 0ULL, 0ULL);
        if (status != CUDA_SUCCESS) {
            WARN(0, "KVcacheSegment reserve ptr failed");
            return;
        }
        this->v_layer_segment_ptr.emplace_back(value_segment_ptr);

        if (device_id == -1) {
            DRV_CALL(cuCtxGetDevice(&device_id));
        }
    }
    if (expand) {
        int expand_num = expand_size / granularitySize;
        this->expandMultiSegment(expand_num);
    }
    
}


bool KVcacheSegment::expandMultiSegment(int expand_num) {
    for (int i = 0; i < this->layer_num; i++) {
        char* k_layer_segment_ptr = (char*)this->k_layer_segment_ptr[i] + this->actual_size;
        char* v_layer_segment_ptr = (char*)this->v_layer_segment_ptr[i] + this->actual_size;
        std::vector<std::shared_ptr<PhyBlock>> k_phy_block_vec;
        std::vector<std::shared_ptr<PhyBlock>> v_phy_block_vec;
        {
            
            if (phy_block_cache.size() >= 2 * expand_num) {
                for (int i = 0; i < expand_num; i++) {
                    k_phy_block_vec.emplace_back(std::move(phy_block_cache[phy_block_cache.size() - 1]));
                    phy_block_cache.pop_back();
                    v_phy_block_vec.emplace_back(std::move(phy_block_cache[phy_block_cache.size() - 1]));
                    phy_block_cache.pop_back();
                }
            }
        }

        if (k_phy_block_vec.size() < expand_num || v_phy_block_vec.size() < expand_num) {
            WARN(0, " KVcacheSegment create PhyBlock failed, phy_block_cache.size:%d, expand_num:%d\n", phy_block_cache.size(), expand_num);
            return false;    
        }

        CUdeviceptr tmp_k_layer_segment_ptr, tmp_v_layer_segment_ptr;
        for (int j = 0; j < expand_num; j++) {
            std::shared_ptr<PhyBlock> k_phy_b = k_phy_block_vec[k_phy_block_vec.size() - 1];
            std::shared_ptr<PhyBlock> v_phy_b = v_phy_block_vec[v_phy_block_vec.size() - 1];
            k_phy_block_vec.pop_back();
            v_phy_block_vec.pop_back();
			CUcontext currentContext;
			CUresult status = cuCtxGetCurrent(&currentContext);
            tmp_k_layer_segment_ptr = reinterpret_cast<CUdeviceptr>(k_layer_segment_ptr + j * granularitySize);
			status = cuMemMap(tmp_k_layer_segment_ptr, granularitySize, 0ULL, k_phy_b->alloc_handle, 0ULL);
            if (status != CUDA_SUCCESS) {
                WARN(0, "expandSegment map failed: %d,",status);
                return false;    
            }
            tmp_v_layer_segment_ptr = reinterpret_cast<CUdeviceptr>(v_layer_segment_ptr + j * granularitySize);
            status = cuMemMap(tmp_v_layer_segment_ptr, granularitySize, 0ULL, v_phy_b->alloc_handle, 0ULL);
            if (status != CUDA_SUCCESS) {
                WARN(0, "expandSegment map failed");
                return false;
            }
            
            CUmemAccessDesc accessDesc = {};
            accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            accessDesc.location.id = this->device_id;
            accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            status = cuMemSetAccess(tmp_k_layer_segment_ptr, granularitySize, &accessDesc, 1);
            if (status != CUDA_SUCCESS) {
                WARN(0, "expandSegment setMemAccess failed");
                return false;
            }
            status = cuMemSetAccess(tmp_v_layer_segment_ptr, granularitySize, &accessDesc, 1);
            if (status != CUDA_SUCCESS) {
                WARN(0, "expandSegment setMemAccess failed");
                return false;
            }

            std::vector<std::shared_ptr<PhyBlock>> k_phy_b_vec;
            std::vector<std::shared_ptr<PhyBlock>> v_phy_b_vec;
            if (this->k_layer_phy_blocks.find(i) != this->k_layer_phy_blocks.end()) {
                k_phy_b_vec = this->k_layer_phy_blocks[i];
                k_phy_b_vec.emplace_back(k_phy_b);
                this->k_layer_phy_blocks[i] = k_phy_b_vec;
            } else {
                k_phy_b_vec.emplace_back(k_phy_b);
                this->k_layer_phy_blocks.insert(std::pair<int, std::vector<std::shared_ptr<PhyBlock>>>(i, k_phy_b_vec));
            }

            if (this->v_layer_phy_blocks.find(i) != this->v_layer_phy_blocks.end()) {
                v_phy_b_vec = this->v_layer_phy_blocks[i];
                v_phy_b_vec.emplace_back(v_phy_b);
                this->v_layer_phy_blocks[i] = v_phy_b_vec;
            } else {
                v_phy_b_vec.emplace_back(v_phy_b);
                this->v_layer_phy_blocks.insert(std::pair<int, std::vector<std::shared_ptr<PhyBlock>>>(i, v_phy_b_vec));
            }
        }
    }
    this->actual_size += expand_num * granularitySize;
    return true;
    
}

CUdeviceptr KVcacheSegment::getKeyDevicePtr(int layer_index) {
    return this->k_layer_segment_ptr[layer_index];
}

CUdeviceptr KVcacheSegment::getValueDevicePtr(int layer_index) {
	    return this->v_layer_segment_ptr[layer_index];
}

void KVcacheSegment::eagerExtend(size_t token_size, int seq_id) {
    size_t curr_kv_offset = this->used_size;
    size_t curr_actual_size = this->actual_size;
    if (this->context_max_size > this->actual_size) {
        int_ptr[seq_id] = 0;
        extend_lock.lock();
        extend_slot.insert(seq_id);
        extendQueue.push(ExtendRequest(this, 1, seq_id, 0));
        extend_lock.unlock();
        slock.lock();
        cycle_ptr[seq_id] = 100;
        slock.unlock();
    }
}

void KVcacheSegment::copy_kv_cache(size_t token_size, size_t kv_size) {
    size_t curr_kv_offset = this->used_size;
    size_t curr_actual_size = this->actual_size;
    if (curr_kv_offset + token_size + 100*kv_size > curr_actual_size) {
        size_t gap_size = curr_kv_offset + token_size + 100*kv_size - curr_actual_size;
        gap_size = granularitySize * ((gap_size + granularitySize - 1) / granularitySize);
        int expand_num = gap_size / granularitySize;

        if (!this->expandMultiSegment(expand_num)) {
            WARN(0, "copy_kv_cache expand segment failed");
            return;
        }
    }

    this->used_size += token_size;
}

uint64_t KVcacheSegment::get_used_slot() {
    return this->used_size;
}

void KVcacheSegment::set_used_size(size_t used_size) {
    this->used_size = used_size;
}


void KVcacheSegment::preempt_seq(int seq_id) {
	extend_lock.lock();
    extend_slot.erase(seq_id);
    extend_lock.unlock();
    
	int_ptr[seq_id] = 1;
	
	slock.lock();
    cycle_ptr[seq_id] = 0;
    slock.unlock();
	
	for (int i = 0; i < this->layer_num; i++) {
        CUdeviceptr k_segment_ptr = this->k_layer_segment_ptr[i];
        CUdeviceptr v_segment_ptr = this->v_layer_segment_ptr[i];
        status = cuMemUnmap(k_segment_ptr, this->actual_size);
        if (status != CUDA_SUCCESS) {
            printf("unmap failed\n %p %d %d", k_segment_ptr, this->actual_size, status);
            return;
        }
		cuMemUnmap(v_segment_ptr, this->actual_size);
        std::vector<std::shared_ptr<PhyBlock>> k_phy_b_vec = this->k_layer_phy_blocks[i];
        std::vector<std::shared_ptr<PhyBlock>> v_phy_b_vec = this->v_layer_phy_blocks[i];
        for (int j = k_phy_b_vec.size() - 1; j >= 0; j--) {
            phy_block_cache.emplace_back(std::move(k_phy_b_vec[j]));
            k_phy_b_vec.pop_back();
            phy_block_cache.emplace_back(std::move(v_phy_b_vec[j]));
            v_phy_b_vec.pop_back();
        }
        this->k_layer_phy_blocks[i] = k_phy_b_vec;
        this->v_layer_phy_blocks[i] = v_phy_b_vec;
    }
    this->used_size = 0;
    this->actual_size = 0;
}

void KVcacheSegment::release_seq(int seq_id) {
    extend_lock.lock();
    extend_slot.erase(seq_id);
    extend_lock.unlock();

    int_ptr[seq_id] = 1;
    
    slock.lock();
    cycle_ptr[seq_id] = 0;
    slock.unlock();
	int res = this->actual_size-this->expand_size;
    int unmap_size = (res >= 0)? res: this->actual_size;
    int unmap_offset = (res >= 0)? this->expand_size: 0;
	if (res != 0) {
	for (int i = 0; i < this->layer_num; i++) {
		CUdeviceptr k_segment_ptr = reinterpret_cast<CUdeviceptr>((char*)this->k_layer_segment_ptr[i] + unmap_offset);
        CUdeviceptr v_segment_ptr = reinterpret_cast<CUdeviceptr>((char*)this->v_layer_segment_ptr[i] + unmap_offset);
		status = cuMemUnmap(k_segment_ptr, unmap_size);
        if (status != CUDA_SUCCESS) {
            printf("unmap failed\n %p %d %d", k_segment_ptr, unmap_size, status);
            return;
        }
        cuMemUnmap(v_segment_ptr, unmap_size);	
		std::vector<std::shared_ptr<PhyBlock>> k_phy_b_vec = this->k_layer_phy_blocks[i];
        std::vector<std::shared_ptr<PhyBlock>> v_phy_b_vec = this->v_layer_phy_blocks[i];
        int n = (this->actual_size - unmap_size) / (2*1024*1024);
        for (int j = k_phy_b_vec.size() - 1; j >= n; j--) {
            phy_block_cache.emplace_back(std::move(k_phy_b_vec[j]));
            k_phy_b_vec.pop_back();
            phy_block_cache.emplace_back(std::move(v_phy_b_vec[j]));
            v_phy_b_vec.pop_back();
        }
        this->k_layer_phy_blocks[i] = k_phy_b_vec;
        this->v_layer_phy_blocks[i] = v_phy_b_vec;
    }
	}
    this->used_size = 0;
    this->actual_size = this->actual_size - unmap_size;
}

void KVcacheSegment::write_string(std::string str) {
    size_t str_size = str.size();
    // layer num is 1
    CUdeviceptr dev_ptr = reinterpret_cast<CUdeviceptr>((char*)this->k_layer_segment_ptr[0] + this->used_size);
    status = cuMemcpyHtoD(dev_ptr, str.c_str(), str_size);
    if (status != CUDA_SUCCESS) {
        printf("write string failed\n");
        return;
    }
    this->used_size += str_size;
}

void KVcacheSegment::read_string() {
    char* str = (char*)malloc(this->used_size * sizeof(char));
    status = cuMemcpyDtoH(str, this->k_layer_segment_ptr[0], this->used_size);
    if (status != CUDA_SUCCESS) {
        printf("read string failed\n");
        return;
    }
    for (int i = 0; i < this->used_size; i++) {
        printf("%c", str[i]);
    }
    printf("\n");
}

std::vector<int> KVcacheSegment::show_info(int seq_id) {
	std::vector<int> res;
	res.push_back(seq_id);
	res.push_back(this->actual_size);
	res.push_back(this->used_size);
	res.push_back(int_ptr[seq_id]);
	res.push_back(cycle_ptr[seq_id]);
	return res;
}

KVCache::KVCache(int num_seqs) {
    this->num_seqs = num_seqs;
    DRV_CALL(cuMemAlloc(&this->kv_cache_ptr, num_seqs * sizeof(uint16_t*)));
}

void KVCache::AddKVCache(int slot, CUdeviceptr kv_token) {
    CUdeviceptr tmp_kv_cache_ptr = reinterpret_cast<CUdeviceptr>((char*)this->kv_cache_ptr + slot * sizeof(uint16_t*));
    DRV_CALL(cuMemcpyHtoD(tmp_kv_cache_ptr, &kv_token, sizeof(uint16_t*)));
}

CUdeviceptr KVCache::getKVCachePtr() {
    return this->kv_cache_ptr;
}

int showFreeBlocks() {
	return phy_block_cache.size(); 
}

