// Copyright 2022 The GLake Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <typeindex>
#include <typeinfo>
#include <type_traits>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <unordered_map>
#include <utility>
#include <iostream>
#include <sstream>
#include <vector>
#include <execinfo.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <stdio.h>


#include <c10/util/intrusive_ptr.h>



typedef unsigned long long CUmemGenericAllocationHandle;

extern int gmlakeInfoLevel = -1;
typedef enum {GMLAKE_LOG_NONE=0, GMLAKE_LOG_INFO=1} gmlakeInfoLogLevel;
pthread_mutex_t gmlakeInfoLock = PTHREAD_MUTEX_INITIALIZER;
FILE* gmlakeInfoFile = stdout;

void gmlakeInfoLog(const char* filefunc, int line, const char* format, ...);

#define GMLAKE_INFO(...) \
	do { \
       gmlakeInfoLog(__func__, __LINE__, __VA_ARGS__); \
	}while(0);
#define gettid() (pid_t) syscall(SYS_gettid)


#define gtrace()  { \
  void *traces[32]; \
  int size = backtrace(traces, 32); \
    char **msgs = backtrace_symbols(traces, size); \
    if (NULL == msgs)  { \
        exit(EXIT_FAILURE); \
    } \
    printf("------------------\n"); \
    int i = 0;for (i = 0; i < size; i++) { \
        printf("[bt] #%d %s symbol:%p \n", i, msgs[i], traces[i]); \
        /* char syscom[256]; sprintf(syscom,"addr2line %p -e /tmp/binomialOptions_kernel", traces[i]); system(syscom);*/ \
        fflush(stdout);\
    } \
    printf("------------------\n"); \
    free (msgs); \
    msgs = NULL; \
}

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
        if(CUDA_SUCCESS == status_val)                                                                   \
        {                                                                                                \
            CUresult result = (call);                                                                    \
            if (CUDA_SUCCESS != result)                                                                  \
            {                                                                                            \
                const char *errMsg; cuGetErrorString(result, &errMsg);                                   \
                WARN(0, "Error when exec " #call " %s-%d code:%d err:%s", __FUNCTION__, __LINE__, result, errMsg); \
            }                                                                                            \
            status_val = result;                                                                         \
        }                                                                                                \
    }




static constexpr size_t granularitySize   =  2097152;

void gmlakeInfoInit() {
    pthread_mutex_lock(&gmlakeInfoLock);
    if (gmlakeInfoLevel != -1) {pthread_mutex_unlock(&gmlakeInfoLock); return;}
    const char* gmlake_info = getenv("GMLAKE_INFO");
    if (gmlake_info == NULL) {
        gmlakeInfoLevel = GMLAKE_LOG_NONE;
    } else if (strcasecmp(gmlake_info, "INFO") == 0) {
        gmlakeInfoLevel = GMLAKE_LOG_INFO;
    }
}

void getHostName(char* hostname, int maxlen, const char delim) {
    if (gethostname(hostname, maxlen) != 0) {
        strncpy(hostname, "unknown", maxlen);
        return;
    }
    int i = 0;
    while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen - 1)) i++;
    hostname[i] = '\0';
    return; 
}

void gmlakeInfoLog(const char* filefunc, int line, const char* fmt, ...) {
    if (gmlakeInfoLevel == -1) gmlakeInfoInit();
    if (gmlakeInfoLevel == GMLAKE_LOG_NONE) return;

    char hostname[1024];
    getHostName(hostname, 1024, '.');
    int cudaDev;
    cudaGetDevice(&cudaDev);
    int pid = getpid();
    int tid = gettid();

    char buffer[1024];
    size_t len = 0;
    len = snprintf(buffer, sizeof(buffer), "%s:%d:%d [%d] GMLAKE_INFO %s():%d", hostname, pid, tid, cudaDev, filefunc, line);
    if (len) {
        va_list vargs;
        va_start(vargs, fmt);
        (void) vsnprintf(buffer+len, sizeof(buffer)-len, fmt, vargs);
        va_end(vargs);
        fprintf(gmlakeInfoFile, "%s\n", buffer);
        fflush(gmlakeInfoFile);
    }

}

size_t getGranularitySize()
{
    static size_t granularity = -1;
  
    if(granularity == -1) {
        int current_device;
        DRV_CALL(cuCtxGetDevice(&current_device));
      
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = current_device;
      
        DRV_CALL(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    }

    return granularity;
}



CUresult setMemAccess(void* ptr, size_t size, int current_device_in = -1)
{
    int current_device = current_device_in;
    if(current_device == -1) {
        DRV_CALL(cuCtxGetDevice(&current_device));
    }

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = current_device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUresult result = cuMemSetAccess((CUdeviceptr)ptr, size, &accessDesc, 1); 
    return result;
}


namespace c10
{
    namespace cuda
    {
        namespace CUDACachingAllocator
        {
            namespace Native
            {   namespace
                {
			        struct Block;
                }
            }
        }
    }
}


struct BlockSegment
{
    BlockSegment():block(nullptr), offset(0) {}
    BlockSegment(c10::cuda::CUDACachingAllocator::Native::Block* block, size_t offset):block(block), offset(offset) {}
    
    c10::cuda::CUDACachingAllocator::Native::Block* block;
    size_t offset;
};

struct PhyBlock
{
    PhyBlock(int device_id_in = -1, size_t block_size_in = granularitySize): 
                                                                            device_id(device_id_in), 
                                                                            block_size(block_size_in), 
                                                                            status(CUDA_SUCCESS), 
                                                                            free(true),
                                                                            owner_stream(nullptr),
                                                                            released(false)
    {
        if(device_id == -1)
        {
          DRV_CALL(cuCtxGetDevice(&device_id));
        }
        
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id;
        
        DRV_CALL_RET(cuMemCreate(&alloc_handle, block_size, &prop, 0), status); 
    }
    
    void release_resources()
    {
        if(status == CUDA_SUCCESS)
        {
            DRV_CALL(cuMemRelease(alloc_handle)); 
        }
        released = true;
    }  
    
    ~PhyBlock()
    {
        if(!released)
        {
            this->release_resources();
            released = true;
        }
    }
    
    int device_id;
    const size_t block_size;
    CUmemGenericAllocationHandle alloc_handle;
    CUresult status;
    
    bool free;
    cudaStream_t owner_stream;
    std::vector<BlockSegment> mapped_blocks;
    bool released;
};



struct VirDevPtr {
    VirDevPtr(CUdeviceptr addr_in, size_t allocSize_in, int device_id = -1): allocSize(allocSize_in), mapped(false), device_id(device_id), status(CUDA_SUCCESS), released(false) {
        if(device_id == -1) {
            DRV_CALL(cuCtxGetDevice(&device_id));
        }
      
        CUdeviceptr device_ptr;
        CUdeviceptr request_ptr = addr_in;
      
        DRV_CALL_RET(cuMemAddressReserve(&device_ptr, allocSize, 0ULL, request_ptr, 0ULL), status);
      
        if(status != CUDA_SUCCESS || (request_ptr != 0ULL && device_ptr != request_ptr)) {
            GMLAKE_INFO(" request_ptr: %p, device_ptr: %p", (void*)request_ptr, (void*)device_ptr);
          
            if(device_ptr != 0ULL) {
                (void)cuMemAddressFree(device_ptr, allocSize);
            }
          
            virAddr = nullptr;
          
            if(status == CUDA_SUCCESS) {
                status = CUDA_ERROR_UNKNOWN;
            }
          
            return;
        }
      
        virAddr = (void*)device_ptr;
    }

    void release_resources() {
      
        if(virAddr) {
            if(mapped) {
                DRV_CALL(cuMemUnmap((CUdeviceptr)virAddr, allocSize));
            }
            DRV_CALL(cuMemAddressFree((CUdeviceptr)virAddr, allocSize)); 
        }
      
        released = true;
    }
  
    ~VirDevPtr() {
        if(!released) {
            this->release_resources();
            released = true;
        }
    }


    void* virAddr;
    const size_t allocSize;
    bool mapped;
    int device_id;
    CUresult status;
    bool released;
};


struct VirBlock
{
    VirBlock(std::shared_ptr<VirDevPtr> vir_dev_ptr_in, 
             size_t offset_in, 
             size_t blockSize_in, 
             std::shared_ptr<PhyBlock> phy_block_in,
             int device_id = -1):vir_dev_ptr(vir_dev_ptr_in),
                                 offset(offset_in), 
                                 blockSize(blockSize_in), 
                                 phy_block(phy_block_in),
                                 device_id(device_id),
                                 status(CUDA_SUCCESS),
                                 released(false) {
        if(device_id == -1) {
          DRV_CALL(cuCtxGetDevice(&device_id));
        }
        
        block_ptr = (void*) ( ((char*)vir_dev_ptr->virAddr) + offset);
        
        CUdeviceptr device_ptr = (CUdeviceptr)block_ptr;
        
        
        DRV_CALL_RET(cuMemMap(device_ptr, blockSize, 0ULL, phy_block->alloc_handle, 0ULL), status);
        DRV_CALL_RET(setMemAccess((void*)device_ptr, blockSize, device_id), status);
        
        if(offset == 0) {
            vir_dev_ptr->mapped = true;
        }
    }

    void release_resources() {
        vir_dev_ptr.reset();
        released = true;
    }
    
    ~VirBlock() {
        if(!released) {
            this->release_resources();
            released = true;
        }
    }
    
    std::shared_ptr<VirDevPtr> vir_dev_ptr;
    
    size_t offset;
    size_t blockSize;
    void* block_ptr;
    
    std::shared_ptr<PhyBlock> phy_block;
    
    int device_id;
    CUresult status;
    bool released;
};


struct VmmSegment {
    VmmSegment():granul_size(0), segment_ptr(nullptr), status(CUDA_SUCCESS), free_blocks(0), released(false) {}
    
    
    VmmSegment(size_t blocks,
               size_t block_size_in = granularitySize,
               int device_id_in = -1):granul_size(block_size_in), 
                                      segment_ptr(nullptr), 
                                      device_id(device_id_in), 
                                      status(CUDA_SUCCESS),
                                      free_blocks(blocks),
                                      used_blocks(0),
                                      fused(false),
                                      released(false) {
        if(device_id == -1) {
          DRV_CALL(cuCtxGetDevice(&device_id));
        }
        
        allocate_phy_blocks(blocks, block_size_in, device_id);
       
        if(status == CUDA_SUCCESS) {
            mapVirAddr();
        }
    }
    
    VmmSegment(std::vector<std::shared_ptr<PhyBlock>>&& phy_blocks_in):phy_blocks(std::move(phy_blocks_in)), 
                                                                          granul_size(phy_blocks[0]->block_size), 
                                                                          segment_ptr(nullptr), 
                                                                          device_id(phy_blocks[0]->device_id),
                                                                          status(CUDA_SUCCESS),
                                                                          free_blocks(phy_blocks.size()),
                                                                          used_blocks(0),
                                                                          fused(true),
                                                                          released(false) {
        mapVirAddr();
    }
    
    
    VmmSegment(std::vector<std::shared_ptr<PhyBlock>> phy_blocks_in, 
               std::vector<std::shared_ptr<VirBlock>> vir_blocks_in):phy_blocks(std::move(phy_blocks_in)), 
                                                                        vir_blocks(std::move(vir_blocks_in)), 
                                                                        granul_size(phy_blocks[0]->block_size), 
                                                                        segment_ptr(vir_blocks[0]->block_ptr),
                                                                        device_id(phy_blocks[0]->device_id), 
                                                                        status(CUDA_SUCCESS),
                                                                        free_blocks(phy_blocks.size()),
                                                                        used_blocks(0),
                                                                        fused(false),
                                                                        released(false)
                                                                        {}

    
    void allocate_phy_blocks(size_t blocks, size_t block_size_in, int device_id_in) {
        
        phy_blocks.reserve(blocks);
        for(size_t i=0; i<blocks; i++)
        {
            auto phy_block = std::make_shared<PhyBlock>(device_id_in, block_size_in);
            if(phy_block->status != CUDA_SUCCESS) {
                size_t device_free;
                size_t device_total;
                cudaMemGetInfo(&device_free, &device_total);
                
                GMLAKE_INFO(" allocate memory handle for %luth phy_block failed, current memory info: device_total: %luMB, device_free: %luMB, request size: %luMB, already allocate: %luMB",
                                             i, device_total/(1024*1024), device_free/(1024*1024), (blocks*block_size_in)/(1024*1024), ((i+1)*block_size_in)/(1024*1024));
      
                
                status = phy_block->status;
                phy_blocks.clear();
                cudaGetLastError();
                break;
            } else {
                phy_blocks.emplace_back(std::move(phy_block));
            }
        }
        
    }
    
    
    void release_resources() {
        
        {
            auto tmp_vir = std::move(vir_blocks);
            auto tmp_phy = std::move(phy_blocks);
        }
        released = true;
    }
    
    virtual ~VmmSegment()
    {
      
        if(!released) {
            this->release_resources();
            released = true;
        }
    }
    
    void* mapVirAddr() {
        static constexpr int retry_times = 8;
        static std::mutex alloc_mutex;

        
        CUdeviceptr device_ptr = 0ULL;
        size_t segment_size = phy_blocks.size() * granul_size;
        
        
        int current_try = 0;
        CUresult result = CUDA_SUCCESS;
        do
        {
            std::lock_guard<std::mutex> lock(alloc_mutex);
            
            auto vir_dev_ptr = std::make_shared<VirDevPtr>(device_ptr, segment_size, device_id);
            
            device_ptr = (CUdeviceptr)vir_dev_ptr->virAddr;
            
            if(vir_dev_ptr->status != CUDA_SUCCESS || !vir_dev_ptr->virAddr) {
                GMLAKE_INFO(" reserve memory of size %fMB failed", segment_size/(1024.f*1024.f));
                
                result = vir_dev_ptr->status;
            } else {
                                
                vir_blocks.clear();
                
                size_t offset = 0;
                for(size_t j = 0; j < phy_blocks.size(); j++) {
                    auto phy_block = phy_blocks[j];
                   
                    auto vir_block = std::make_shared<VirBlock>(vir_dev_ptr, offset, granul_size, phy_block, device_id);
                    
                    if(vir_block->status != CUDA_SUCCESS) {
                        result = vir_block->status;
                        vir_blocks.clear();
                        cudaGetLastError();
                        GMLAKE_INFO(" map memory %p of size %fMB for the %luth phy_block failed", vir_block->block_ptr, granul_size/(1024.f*1024.f), j);
                        break;
                    } else {
                        vir_blocks.emplace_back(std::move(vir_block));
                    }
                    
                    offset += granul_size;
                }
                
            }
            
            current_try++;
            device_ptr = 0ULL;
        } while(result != CUDA_SUCCESS && current_try < retry_times);    
        
        
        status = result;
        
        
        if(result == CUDA_SUCCESS) {
            segment_ptr = vir_blocks[0]->block_ptr;
            return segment_ptr;
        }
        
        
        return nullptr;
    }
    
    
    
    
    std::shared_ptr<VmmSegment> split(size_t keep_size) {
        if(keep_size%granul_size) {
            GMLAKE_INFO(" keep_size %fMB is not multiple of granul_size %fMB", keep_size/(1024.f*1024.f), granul_size/(1024.f*1024.f));
            gtrace();
            exit(-1);
        }
        
        size_t keep_blocks = keep_size/granul_size;
        
        
        if(keep_blocks >= vir_blocks.size()) {
            GMLAKE_INFO(" keep_blocks %lu is larger than remaing blocks %lu", keep_blocks, vir_blocks.size());
            gtrace();
            exit(-1);
        }

        
        std::vector<std::shared_ptr<PhyBlock>> remain_phy_blocks;
        std::vector<std::shared_ptr<VirBlock>> remain_vir_blocks;
        
        size_t remaining_free_blocks = 0;
        for(size_t i=keep_blocks; i<phy_blocks.size(); i++) {
            if(phy_blocks[i]->free) {
                remaining_free_blocks++;
            }
            
            remain_phy_blocks.emplace_back(std::move(phy_blocks[i]));
            remain_vir_blocks.emplace_back(std::move(vir_blocks[i]));
        }
        
        this->phy_blocks.resize(keep_blocks);
        this->vir_blocks.resize(keep_blocks);
        
        auto remaining_segment = std::make_shared<VmmSegment>(std::move(remain_phy_blocks), std::move(remain_vir_blocks));
        
        remaining_segment->segment_ptr = (void*)( (char*)segment_ptr + keep_size );
        remaining_segment->free_blocks = remaining_free_blocks;

        free_blocks -= remaining_free_blocks;
        
        return remaining_segment;
    }
    
    
    bool remerge(VmmSegment& segment) {
        if( segment.segment_ptr ==  (void*) ( (char*)this->segment_ptr + this->phy_blocks.size()*granul_size) ) {
           for(size_t i=0; i< segment.phy_blocks.size(); i++) {
               this->phy_blocks.emplace_back(std::move(segment.phy_blocks[i]));
               this->vir_blocks.emplace_back(std::move(segment.vir_blocks[i]));
           }
        } else if( this->segment_ptr == (void*) ( (char*)segment.segment_ptr + segment.phy_blocks.size()*granul_size) ) {
           for(size_t i=0; i< phy_blocks.size(); i++) {
               segment.phy_blocks.emplace_back(std::move(this->phy_blocks[i]));
               segment.vir_blocks.emplace_back(std::move(this->vir_blocks[i]));
            }
           
           this->phy_blocks = std::move(segment.phy_blocks);
           this->vir_blocks = std::move(segment.vir_blocks);
           
           this->segment_ptr = segment.segment_ptr;
        } else {
            GMLAKE_INFO(" segment of ptr %p size %fMB is not head or tail of segment ptr %p size %fMB", 
                                                                      segment.segment_ptr, segment.vir_blocks.size()*granul_size/(1024.f*1024.f),
                                                                      this->segment_ptr, this->vir_blocks.size()*granul_size/(1024.f*1024.f));
            exit(-1);
            
           return false;
        }
        
        
        this->free_blocks += segment.free_blocks;
        segment.free_blocks = 0;
        
        segment.phy_blocks.clear();
        segment.vir_blocks.clear();
        
        segment.segment_ptr = nullptr;
        
        return true;
    }
    
    
    std::vector<std::shared_ptr<PhyBlock>> phy_blocks;
    std::vector<std::shared_ptr<VirBlock>> vir_blocks;
    
    
    const size_t granul_size;
    void* segment_ptr;
    
    int device_id;
    CUresult status;
    
    size_t free_blocks;
    size_t used_blocks;
    bool fused;
    bool released;
};

