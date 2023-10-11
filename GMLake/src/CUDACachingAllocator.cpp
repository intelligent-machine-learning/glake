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

#include <c10/cuda/CUDACachingAllocator.h>

#include <c10/core/impl/GPUTrace.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/util/llvmMathExtras.h>
#include <c10/util/Backtrace.h>
#include <unordered_map>
#include <unordered_set>
#include <c10/cuda/cuda_vmm_allocator.h>

#include <cuda_runtime_api.h>
#include <algorithm>
#include <bitset>
#include <cstdint>
#include <deque>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <utility>
#include <vector>

namespace c10 {

C10_DEFINE_REGISTRY(FreeCudaMemoryCallbacksRegistry, FreeMemoryCallback);

namespace cuda {
namespace CUDACachingAllocator {
namespace Native {

//
// Yet another caching allocator for CUDA device allocations.
//
// - Allocations are associated with a stream. Once freed, blocks can be
//   re-allocated on the same stream, but not on any other stream.
// - The allocator attempts to find the smallest cached block that will fit the
//   requested size. If the block is larger than the requested size, it may be
//   split. If no block is found, the allocator will delegate to cudaMalloc.
// - If the cudaMalloc fails, the allocator will attempt to free one cached
//   block of sufficient size that is not split and retry the allocation.
//   If this also fails, the allocator will attempt to free all cached blocks
//   that are not split and retry the allocation.
// - Large (>1MB) and small allocations are stored in separate pools.
//   Small requests are packed into 2MB buffers. Large requests will use the
//   smallest available free block or allocate a new block using cudaMalloc.
// - To reduce fragmentation, requests between 1MB and 10MB will allocate and
//   split a 20MB block, if no free block of sufficient size is available.
// - To further reduce fragmentation, blocks >= 200MB are not allowed to be
//   split. These oversize cached blocks will still satisfy requests within
//   20MB of the oversize cached block size.
//
// With this allocator, allocations and frees should logically be considered
// "usages" of the memory segment associated with streams, just like kernel
// launches. The programmer must insert the proper synchronization if memory
// segments are used from multiple streams.
//
// The library provides a recordStream() function to help insert the correct
// synchronization when allocations are used on multiple streams. This will
// ensure that the block is not reused before each recorded stream completes
// work.
//

/**
 * Note [Interaction with CUDA graph capture]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Graph capture performs a dry run of a region of execution, freezing all CUDA
 * work (and virtual addresses used during that work) into a "graph." The graph
 * may be "replayed" like a single giant kernel, with greatly reduced CPU
 * overhead as well as modestly improved GPU performance.
 *
 * Because capture bakes in memory addresses, the memory used during capture
 * must be available for the graph to use during replay. DeviceCachingAllocator
 * assigns and frees memory eagerly and dynamically, so if we're not careful
 * about managing graphs' memory, at replay time those memory addresses could be
 * used by other tensors.
 *
 * To guarantee a graph's baked in addresses are safe to reuse in replay,
 * DeviceAllocator satisfies allocations from a graph-private memory pool during
 * capture, and doesn't begin cudaFreeing those addresses until the graph is
 * destroyed.
 *
 * Within the private pool, allocations are freed and reassigned as usual during
 * capture. Memory regions will be used in a consistent order during replay. So
 * a private pool doesn't use memory more wastefully than the default pools
 * during capture, but it does reserve its high-water mark of used memory away
 * from the default pools as long as the capture(s) it served survive
 * (regardless whether those captures are idle or replaying).
 *
 * CUDAGraph's requests for private pools are mediated by
 * DeviceAllocator::notifyCaptureBegin,
 *                  notifyCaptureAboutToEnd,
 *                  notifyCaptureEnded,
 *                  notifyCaptureDestroy.
 */

constexpr size_t kMinBlockSize =
    512; // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576; // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer =
    2097152; // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer =
    20971520; // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc =
    10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152; // round up large allocations to 2 MiB
constexpr size_t kGranularity   =  2097152; // round up large allocations to 2 MiB
constexpr size_t kRoundUpPowerOfTwoIntervals = 16;

namespace {

using stream_set = ska::flat_hash_set<cuda::CUDAStream>;

using StatTypes = std::array<bool, static_cast<size_t>(StatType::NUM_TYPES)>;

void update_stat(Stat& stat, int64_t amount) {
  stat.current += amount;

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stat.current >= 0,
      "Negative tracked stat in CUDA allocator (likely logic error).");

  stat.peak = std::max(stat.current, stat.peak);
  if (amount > 0) {
    stat.allocated += amount;
  }
  if (amount < 0) {
    stat.freed += -amount;
  }
}

void reset_accumulated_stat(Stat& stat) {
  stat.allocated = 0;
  stat.freed = 0;
}

void reset_peak_stat(Stat& stat) {
  stat.peak = stat.current;
}

template <typename Func>
void for_each_selected_stat_type(const StatTypes& stat_types, Func f) {
  for (const auto stat_type : c10::irange(stat_types.size())) {
    if (stat_types[stat_type]) {
      f(stat_type);
    }
  }
}

void update_stat_array(
    StatArray& stat_array,
    int64_t amount,
    const StatTypes& stat_types) {
  for_each_selected_stat_type(
      stat_types, [&stat_array, amount](size_t stat_type) {
        update_stat(stat_array[stat_type], amount);
      });
}


struct EventIDCounter
{
  EventIDCounter(cudaStream_t stream):stream(stream), current_event_id(0) {}
    
  void reset() {
    std::lock_guard<std::recursive_mutex> lock(id_mutex);
    current_event_id = 0;
  }
    
  std::uint64_t next_id() {
    std::lock_guard<std::recursive_mutex> lock(id_mutex);
    
    if(current_event_id == std::numeric_limits<uint64_t>::max())
      current_event_id = 1;
    else
      current_event_id++;
        
      return current_event_id;
  }
    
  std::recursive_mutex id_mutex;
  cudaStream_t stream;
  std::uint64_t current_event_id;
};

static std::unordered_map<cudaStream_t, std::shared_ptr<EventIDCounter>> stream_id_counter;
static std::mutex counter_mutex;
struct BlockEvent {
  BlockEvent(cudaStream_t stream_in, bool record_event=false) {
    stream = stream_in;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    event_id = 0;
    released = false;
    ref_as_sync = false;
    if(record_event) record(stream);
    
  }

  void record(cudaStream_t stream_in) {

    if(stream == stream_in)
    {
      std::shared_ptr<EventIDCounter> id_counter;
      if(stream_id_counter.find(stream) == stream_id_counter.end()) {
        id_counter = std::make_shared<EventIDCounter>(stream);
        {
          std::lock_guard<std::mutex> lock(counter_mutex);
          stream_id_counter[stream] = id_counter;
        }
      } else {
        std::lock_guard<std::mutex> lock(counter_mutex);
        id_counter = stream_id_counter[stream];
      }
      
      {
        std::lock_guard<std::recursive_mutex> lock(id_counter->id_mutex);
        
        event_id = id_counter->next_id();
        C10_CUDA_CHECK(cudaEventRecord(event, stream));
      }
    }
  }

  void release_resources()
  {
    if(!ref_as_sync) {
      C10_CUDA_CHECK(cudaEventDestroy(event));
    } else {
      cudaError_t err = cudaEventQuery(event);
      if(err == cudaSuccess) {
        C10_CUDA_CHECK(cudaEventDestroy(event));
      } else if(err == cudaErrorNotReady) {
        cudaGetLastError();
        event_gc(stream, event_id, event);
      } else {
        C10_CUDA_CHECK(err);
        cudaGetLastError();
        C10_CUDA_CHECK(cudaEventDestroy(event));
      }
    }
  }

  ~BlockEvent() {
    if (!released) {
      this->release_resources();
      released = true;
    }
  }

  static uint64_t create_id()
  {
    static std::atomic<std::uint64_t> current_event_id = {0};
    static uint64_t id_max = std::numeric_limits<uint64_t>::max();

    current_event_id.compare_exchange_strong(id_max, std::uint64_t(0));
    uint64_t my_id = current_event_id.fetch_add(1);
    
    return my_id;
  }

  static void event_gc(cudaStream_t stream, uint64_t event_id, cudaEvent_t event)
  {
    using EventQueue=std::map<uint64_t, cudaEvent_t>;
    
    static std::unordered_map<cudaStream_t, EventQueue> event_pool;
    static std::mutex pool_mutex;

    {
      std::lock_guard<std::mutex> lock(pool_mutex);
      event_pool[stream][event_id] = event;
    }
    
    auto& event_queue = event_pool.at(stream);
    if(event_queue.size() > 2000)
    {
      std::lock_guard<std::mutex> lock(pool_mutex);

      for(auto it = event_queue.begin(); it != std::prev(event_queue.end());) {
        cudaEvent_t event = it->second;
        cudaError_t err = cudaEventQuery(event);
        if(err == cudaSuccess) {
          C10_CUDA_CHECK(cudaEventDestroy(event));
          it = event_queue.erase(it);
        } else {
          cudaGetLastError();
          break;
        }
      }
    }
  }
  
  cudaStream_t stream;
  cudaEvent_t event;
  uint64_t event_id;
  bool released;
  bool ref_as_sync;
};


struct Block;
struct PrivatePool;
typedef bool (*Comparison)(const Block*, const Block*);

struct BlockPool {
  BlockPool(
      Comparison comparator,
      bool small,
      PrivatePool* private_pool = nullptr)
      : blocks(comparator), is_small(small), owner_PrivatePool(private_pool) {}
  std::set<Block*, Comparison> blocks;
  const bool is_small;
  PrivatePool* owner_PrivatePool;
};

struct HistoryChain {
  History h;
  std::unique_ptr<HistoryChain> next; // when blocks are merged we keep records
                                      // of what used to be in the block
};

struct Block {
  int device; // gpu
  cudaStream_t stream; // allocation stream
  stream_set stream_uses; // streams on which the block was used
  size_t size; // block size in bytes
  size_t requested_size; // memory originally requested
  size_t actual_size;
  BlockPool* pool{nullptr}; // owning memory pool
  void* ptr{nullptr}; // memory address
  bool allocated{false}; // in-use flag
  Block* prev{nullptr}; // prev block if split from a larger allocation
  Block* next{nullptr}; // next block if split from a larger allocation
  int event_count{0}; // number of outstanding CUDA events
  int gc_count{0}; // counter for prioritizing older / less useful blocks for
                   // garbage collection
  std::unique_ptr<HistoryChain> history;
  HistoryChain* history_last{nullptr};
  std::shared_ptr<VmmSegment> vmm_segment;
  std::shared_ptr<BlockEvent> self_last_event;

  Block(
      int device,
      cudaStream_t stream,
      size_t size,
      BlockPool* pool,
      void* ptr)
      : device(device),
        stream(stream),
        stream_uses(),
        size(size),
        actual_size(0),
        requested_size(0),
        pool(pool),
        self_last_event(std::make_shared<BlockEvent>(stream)),
        ptr(ptr) {}

  // constructor for search key
  Block(int device, cudaStream_t stream, size_t size)
      : device(device),
        stream(stream),
        stream_uses(),
        size(size),
        actual_size(0),
        self_last_event(std::make_shared<BlockEvent>(stream)),
        requested_size(0) {}

  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }
};

static bool BlockComparator(const Block* a, const Block* b) {
  if (a->stream != b->stream) {
    return (uintptr_t)a->stream < (uintptr_t)b->stream;
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

struct BlockEventOrderComparator {
  using BlockPtr=Block*;

  bool operator()(const BlockPtr a, const BlockPtr b) const {
    if(!a->self_last_event && !b->self_last_event) {
      if(a->size != b->size) {
        return a->size < b->size;
      }
  
      return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
    }
    
    if(!a->self_last_event) {
      return true;
    }
  
    if(!b->self_last_event) {
      return false;
    }


    if(a->self_last_event->event_id != b->self_last_event->event_id) {
        return a->self_last_event->event_id < b->self_last_event->event_id;
    }

    
    if(a->size != b->size) {
      return a->size < b->size;
    }
  
    return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
  }
};

using EventOrderedBlockSet=std::set<Block*, BlockEventOrderComparator>;
using SetIterator=EventOrderedBlockSet::iterator;

struct BlockEventOrderPool
{
  BlockEventOrderPool():pool_size(0) {}
    
  void insert(Block* block) {
    if(blocks.count(block) == 0) {
      blocks.insert(block);
      pool_size += block->size;
    }
  }
    
  bool erase(Block* block) {
    if(blocks.count(block)) {
      blocks.erase(block);
      pool_size -= block->size;
            
      return true;
    } else {
      GMLAKE_INFO(" warning block %p, block ptr %p of size %lu not found in pool", block, block->ptr, block->size);
      return false;
    }
  }

  SetIterator erase(SetIterator it) {
    if(blocks.count(*it)) {
      pool_size -= (*it)->size;
            
      return blocks.erase(it);
    } else {
      GMLAKE_INFO(" warning block %p, block ptr %p of size %lu not found in pool", (*it), (*it)->ptr, (*it)->size);
      return blocks.end();
    }
  }
    
    
  EventOrderedBlockSet blocks;
  size_t pool_size;
};

struct AllocParams {
  AllocParams(
      int device,
      size_t size,
      cudaStream_t stream,
      BlockPool* pool,
      size_t alloc_size,
      DeviceStats& stats)
      : search_key(device, stream, size),
        pool(pool),
        alloc_size(alloc_size),
        block(nullptr),
        err(cudaSuccess) {}

  int device() const {
    return search_key.device;
  }
  cudaStream_t stream() const {
    return search_key.stream;
  }
  size_t size() const {
    return search_key.size;
  }

  Block search_key;
  BlockPool* pool;
  size_t alloc_size;
  Block* block;
  StatTypes stat_types = {false};
  cudaError_t err;
};

int trimHistoryBefore(Block* block, void* point) {
  int n = 0;
  while (block->history && block->history->h.addr < point) {
    block->history = std::move(block->history->next);
    ++n;
  }
  if (!block->history) {
    block->history_last = nullptr;
  }
  return n;
}

// Note: cudaEventCreate when concurrently invoked from multiple threads can be
// very expensive (at least on certain device/driver combinations). Thus, we a)
// serialize event creation at a per-device level, and b) pool the events to
// avoid constantly calling cudaEventCreate/cudaEventDestroy. This results in
// significant improvements in multithreaded workloads with high allocation
// rates.
class EventPool {
 public:
  using Event = std::unique_ptr<cudaEvent_t, std::function<void(cudaEvent_t*)>>;
  // TODO: Explicit device count
  EventPool() : pools_(at::cuda::device_count()) {}

  Event get(int device) {
    TORCH_INTERNAL_ASSERT(0 <= device);
    TORCH_INTERNAL_ASSERT(device < static_cast<int>(pools_.size()));
    auto& pool = pools_[device];
    auto destructor = [&pool](cudaEvent_t* event) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<cudaEvent_t>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
      std::lock_guard<std::mutex> g(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto* event = pool.event_pool_.back().release();
        pool.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }
    // otherwise, allocate a new event that will be returned to the pool on
    // destruction.
    auto new_ptr = std::make_unique<cudaEvent_t>();
    C10_CUDA_CHECK(
        cudaEventCreateWithFlags(new_ptr.get(), cudaEventDisableTiming));

    return Event(new_ptr.release(), destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  struct PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::vector<std::unique_ptr<cudaEvent_t>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
};

// CUDA graphs helper
struct PrivatePool {
  PrivatePool()
      : use_count(1),
        cudaMalloc_count(0),
        large_blocks(BlockComparator, /*is_small=*/false, this),
        small_blocks(BlockComparator, /*is_small=*/true, this) {}
  PrivatePool(const PrivatePool&) = delete;
  PrivatePool(PrivatePool&&) = delete;
  PrivatePool& operator=(const PrivatePool&) = delete;
  // Number of live graphs using this pool
  int use_count;
  // Number of unfreed cudaMallocs made for this pool. When use_count and
  // cudaMalloc_count drop to zero, we can delete this PrivatePool from
  // graph_pools.
  int cudaMalloc_count;
  // Instead of maintaining private BlockPools here, I could stuff all blocks
  // (private or no) into the top-level large_blocks and small_blocks, and
  // distinguish private blocks by adding a "pool id" check above the stream
  // check in BlockComparator. BlockComparator is performance- critial though,
  // I'd rather not add more logic to it.
  BlockPool large_blocks;
  BlockPool small_blocks;
};

struct MempoolIdHash {
  std::size_t operator()(const MempoolId_t& mempool_id) const noexcept {
    return mempool_id.first != 0 ? mempool_id.first : mempool_id.second;
  }
};

cudaError_t cudaMallocMaybeCapturing(void** p, size_t size) {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  if (at::cuda::currentStreamCaptureStatusMayInitCtx() ==
      at::cuda::CaptureStatus::None) {
#endif
    return C10_CUDA_ERROR_HANDLED(cudaMalloc(p, size));
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  } else {
    // It's ok to capture cudaMallocs, as long as we never cudaFree those
    // addresses before replay.
    // Capturing cudaMalloc behaves nicely: it gives the graph new VA,
    // but is ignored (won't leakily allocate new memory) in replays.
    at::cuda::CUDAStreamCaptureModeGuard g{cudaStreamCaptureModeRelaxed};
    return C10_CUDA_ERROR_HANDLED(cudaMalloc(p, size));
  }
#endif
}

} // anonymous namespace
} // namespace Native

// Environment config parser
// Defined here, rather than its own .cpp file,
// because parseArgs needs to know kLargeBuffer.
// Defined outside namespace Native because it's not Native-specific.
class CachingAllocatorConfig {
 public:
  static size_t max_split_size() {
    return instance().m_max_split_size;
  }
  static double garbage_collection_threshold() {
    return instance().m_garbage_collection_threshold;
  }

  // This is used to round-up allocation size to nearest power of 2 divisions.
  // More description below in function roundup_power2_next_division
  // As ane example, if we want 4 divisions between 2's power, this can be done
  // using env variable: PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:4
  static size_t roundup_power2_divisions(size_t size) {
    size_t log_size = (63 - llvm::countLeadingZeros(size));

    // Our intervals start at 1MB and end at 64GB
    const size_t interval_start =
        63 - llvm::countLeadingZeros(static_cast<size_t>(1048576));
    const size_t interval_end =
        63 - llvm::countLeadingZeros(static_cast<size_t>(68719476736));
    TORCH_CHECK(
        (interval_end - interval_start == Native::kRoundUpPowerOfTwoIntervals),
        "kRoundUpPowerOfTwoIntervals mismatch");

    int index = static_cast<int>(log_size) - static_cast<int>(interval_start);

    index = std::max(0, index);
    index = std::min(
        index, static_cast<int>(Native::kRoundUpPowerOfTwoIntervals) - 1);
    return instance().m_roundup_power2_divisions[index];
  }

  static CachingAllocatorConfig& instance() {
    static CachingAllocatorConfig* s_instance = ([]() {
      auto inst = new CachingAllocatorConfig();
      const char* env = getenv("PYTORCH_CUDA_ALLOC_CONF");
      inst->parseArgs(env);
      return inst;
    })();
    return *s_instance;
  }

  void parseArgs(const char* env);

 private:
  CachingAllocatorConfig()
      : m_max_split_size(std::numeric_limits<size_t>::max()),
        m_garbage_collection_threshold(0) {
    m_roundup_power2_divisions.assign(Native::kRoundUpPowerOfTwoIntervals, 0);
  }

  void lexArgs(const char* env, std::vector<std::string>& config);
  void consumeToken(
      const std::vector<std::string>& config,
      size_t i,
      const char c);
  size_t parseMaxSplitSize(const std::vector<std::string>& config, size_t i);
  size_t parseGarbageCollectionThreshold(
      const std::vector<std::string>& config,
      size_t i);
  size_t parseRoundUpPower2Divisions(
      const std::vector<std::string>& config,
      size_t i);
  size_t parseAllocatorConfig(
      const std::vector<std::string>& config,
      size_t i,
      bool& used_cudaMallocAsync);

  std::atomic<size_t> m_max_split_size;
  std::vector<size_t> m_roundup_power2_divisions;
  std::atomic<double> m_garbage_collection_threshold;
};

void CachingAllocatorConfig::lexArgs(
    const char* env,
    std::vector<std::string>& config) {
  std::vector<char> buf;

  size_t env_length = strlen(env);
  for (size_t i = 0; i < env_length; i++) {
    if (env[i] == ',' || env[i] == ':' || env[i] == '[' || env[i] == ']') {
      if (buf.size() != 0) {
        config.emplace_back(buf.begin(), buf.end());
        buf.clear();
      }
      config.emplace_back(1, env[i]);
    } else if (env[i] != ' ') {
      buf.emplace_back(static_cast<char>(env[i]));
    }
  }
  if (!buf.empty()) {
    config.emplace_back(buf.begin(), buf.end());
  }
}

void CachingAllocatorConfig::consumeToken(
    const std::vector<std::string>& config,
    size_t i,
    const char c) {
  TORCH_CHECK(
      i < config.size() && config[i].compare(std::string(1, c)) == 0,
      "Error parsing CachingAllocator settings, expected ",
      c,
      "");
}

size_t CachingAllocatorConfig::parseMaxSplitSize(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    size_t val1 = stoi(config[i]);
    TORCH_CHECK(
        val1 > Native::kLargeBuffer / (1024 * 1024),
        "CachingAllocator option max_split_size_mb too small, must be > ",
        Native::kLargeBuffer / (1024 * 1024),
        "");
    val1 = std::max(val1, Native::kLargeBuffer / (1024 * 1024));
    val1 = std::min(val1, (std::numeric_limits<size_t>::max() / (1024 * 1024)));
    m_max_split_size = val1 * 1024 * 1024;
  } else {
    TORCH_CHECK(false, "Error, expecting max_split_size_mb value", "");
  }
  return i;
}

size_t CachingAllocatorConfig::parseGarbageCollectionThreshold(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    double val1 = stod(config[i]);
    TORCH_CHECK(
        val1 > 0, "garbage_collect_threshold too small, set it 0.0~1.0", "");
    TORCH_CHECK(
        val1 < 1.0, "garbage_collect_threshold too big, set it 0.0~1.0", "");
    m_garbage_collection_threshold = val1;
  } else {
    TORCH_CHECK(
        false, "Error, expecting garbage_collection_threshold value", "");
  }
  return i;
}

size_t CachingAllocatorConfig::parseRoundUpPower2Divisions(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  bool first_value = true;

  if (++i < config.size()) {
    if (config[i].compare("[") == 0) {
      size_t last_index = 0;
      while (++i < config.size() && config[i].compare("]") != 0) {
        std::string val1 = config[i];
        size_t val2 = 0;

        consumeToken(config, ++i, ':');
        if (++i < config.size()) {
          val2 = stoi(config[i]);
        } else {
          TORCH_CHECK(
              false, "Error parsing roundup_power2_divisions value", "");
        }
        TORCH_CHECK(
            llvm::isPowerOf2_64(val2),
            "For roundups, the divisons has to be power of 2 ",
            "");

        if (val1.compare(">") == 0) {
          std::fill(
              std::next(
                  m_roundup_power2_divisions.begin(),
                  static_cast<std::vector<unsigned long>::difference_type>(
                      last_index)),
              m_roundup_power2_divisions.end(),
              val2);
        } else {
          size_t val1_long = stoul(val1);
          TORCH_CHECK(
              llvm::isPowerOf2_64(val1_long),
              "For roundups, the intervals have to be power of 2 ",
              "");

          size_t index = 63 - llvm::countLeadingZeros(val1_long);
          index = std::max((size_t)0, index);
          index = std::min(index, m_roundup_power2_divisions.size() - 1);

          if (first_value) {
            std::fill(
                m_roundup_power2_divisions.begin(),
                std::next(
                    m_roundup_power2_divisions.begin(),
                    static_cast<std::vector<unsigned long>::difference_type>(
                        index)),
                val2);
            first_value = false;
          }
          if (index < m_roundup_power2_divisions.size()) {
            m_roundup_power2_divisions[index] = val2;
          }
          last_index = index;
        }

        if (config[i + 1].compare("]") != 0) {
          consumeToken(config, ++i, ',');
        }
      }
    } else { // Keep this for backwards compatibility
      size_t val1 = stoi(config[i]);
      TORCH_CHECK(
          llvm::isPowerOf2_64(val1),
          "For roundups, the divisons has to be power of 2 ",
          "");
      std::fill(
          m_roundup_power2_divisions.begin(),
          m_roundup_power2_divisions.end(),
          val1);
    }
  } else {
    TORCH_CHECK(false, "Error, expecting roundup_power2_divisions value", "");
  }
  return i;
}

size_t CachingAllocatorConfig::parseAllocatorConfig(
    const std::vector<std::string>& config,
    size_t i,
    bool& used_cudaMallocAsync) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    TORCH_CHECK(
        ((config[i] == "native") || (config[i] == "cudaMallocAsync")),
        "Unknown allocator backend, "
        "options are native and cudaMallocAsync");
    used_cudaMallocAsync = (config[i] == "cudaMallocAsync");
    if (used_cudaMallocAsync) {
#if CUDA_VERSION >= 11040
      int version;
      C10_CUDA_CHECK(cudaDriverGetVersion(&version));
      TORCH_CHECK(
          version >= 11040,
          "backend:cudaMallocAsync requires CUDA runtime "
          "11.4 or newer, but cudaDriverGetVersion returned ",
          version);
#else
      TORCH_CHECK(
          false,
          "backend:cudaMallocAsync requires PyTorch to be built with "
          "CUDA 11.4 or newer, but CUDA_VERSION is ",
          CUDA_VERSION);
#endif
    }
    TORCH_INTERNAL_ASSERT(
        config[i] == get()->name(),
        "Allocator backend parsed at runtime != "
        "allocator backend parsed at load time");
  } else {
    TORCH_CHECK(false, "Error parsing backend value", "");
  }
  return i;
}

void CachingAllocatorConfig::parseArgs(const char* env) {
  // If empty, set the default values
  m_max_split_size = std::numeric_limits<size_t>::max();
  m_roundup_power2_divisions.assign(Native::kRoundUpPowerOfTwoIntervals, 0);
  m_garbage_collection_threshold = 0;
  bool used_cudaMallocAsync = false;
  bool used_native_specific_option = false;

  if (env == nullptr) {
    return;
  }

  std::vector<std::string> config;
  lexArgs(env, config);

  for (size_t i = 0; i < config.size(); i++) {
    if (config[i].compare("max_split_size_mb") == 0) {
      i = parseMaxSplitSize(config, i);
      used_native_specific_option = true;
    } else if (config[i].compare("garbage_collection_threshold") == 0) {
      i = parseGarbageCollectionThreshold(config, i);
      used_native_specific_option = true;
    } else if (config[i].compare("roundup_power2_divisions") == 0) {
      i = parseRoundUpPower2Divisions(config, i);
      used_native_specific_option = true;
    } else if (config[i].compare("backend") == 0) {
      i = parseAllocatorConfig(config, i, used_cudaMallocAsync);
    } else {
      TORCH_CHECK(false, "Unrecognized CachingAllocator option: ", config[i]);
    }

    if (i + 1 < config.size()) {
      consumeToken(config, ++i, ',');
    }
  }

  if (used_cudaMallocAsync && used_native_specific_option) {
    TORCH_WARN(
        "backend:cudaMallocAsync ignores max_split_size_mb, roundup_bypass_threshold_mb,"
        "roundup_power2_divisions, and garbage_collect_threshold.");
  }
}

namespace Native {

class DeviceCachingAllocator {
 private:
  // lock around all operations
  mutable std::recursive_mutex mutex;

  // device statistics
  DeviceStats stats;

  // unallocated cached blocks larger than 1 MB
  BlockPool large_blocks;

    
  // unallocated cached blocks larger than 64 MB
  //BlockPool huge_blocks;

  // fused blocks that has been mapped to fragment blocks in size order
  BlockPool free_fused_blocks;
  
  // fused blocks that has been mapped to fragment blocks in release order
  std::unordered_map<cudaStream_t, BlockEventOrderPool> free_fused_blocks_in_release_order;
  
  // fused blocks which is free, but it's phy_blocks are used by other block of my stream
  std::unordered_map<cudaStream_t, BlockEventOrderPool> fragmented_free_fused_blocks;

  // unallocated cached blocks 1 MB or smaller
  BlockPool small_blocks;

  // allocated or in use by a stream. Holds all active allocations,
  // whether they came from graph_pools or one of the BlockPools above.
  ska::flat_hash_set<Block*> active_blocks;
  
    //active fused blocks
  ska::flat_hash_set<Block*> active_fused_blocks;
  
  //active fused blocks to be garbage collected
  ska::flat_hash_set<Block*> active_fused_blocks_to_gc;

  // captures_underway tracks if a capture might be underway on any stream.
  // Most of the time it's zero, in which case malloc can avoid calling
  // cudaStreamGetCaptureInfo in the hot path.
  int captures_underway = 0;
  // See free() for this thing's purpose
  std::vector<Block*> needs_events_deferred_until_no_capture;
  // outstanding cuda events
  ska::flat_hash_map<
      cuda::CUDAStream,
      std::deque<std::pair<EventPool::Event, Block*>>>
      cuda_events;

  // record used memory.
  size_t total_allocated_memory = 0;

  size_t total_fuse_size = 0;

  size_t allowed_memory_maximum = 0;

  bool set_fraction = false;

  bool record_history = false;
  std::atomic<CreateContextFn> context_recorder_;
  size_t alloc_trace_next = 0;
  bool alloc_trace_record_context_ = false;
  size_t alloc_trace_max_entries_ = 1;
  std::vector<TraceEntry>*
      alloc_trace; // pointer because we need to intentionally leak this on
                   // deallocation it can hold references to Python state which
                   // will already be destroyed when we are in exit handlers

  // Members specific to CUDA graphs

  // Private pools for CUDA graphs
  ska::flat_hash_map<MempoolId_t, std::unique_ptr<PrivatePool>, MempoolIdHash>
      graph_pools;
  // Pools no longer referenced by any graph. Their BlockPools are eligible for
  // free_blocks. Can't be a vector or deque because we might erase entries in
  // any order. Could be an std::list, but we don't care much, access and
  // insert/erase are rare.
  ska::flat_hash_map<MempoolId_t, PrivatePool*, MempoolIdHash>
      graph_pools_freeable;

  // Maps a capturing stream to its assigned private pool,
  // in case we want multiple captures to share the same pool
  ska::flat_hash_map<CaptureId_t, MempoolId_t> capture_to_pool_map;

  // XXX - maybe we should generalize and have multiple events
  std::vector<OutOfMemoryObserver> oom_observers_;

 public:
  DeviceCachingAllocator()
      : large_blocks(BlockComparator, /*is_small=*/false),
        free_fused_blocks(BlockComparator, /*is_small=*/false),
        small_blocks(BlockComparator, /*is_small=*/true),
        alloc_trace(new std::vector<TraceEntry>()) {
    stats.max_split_size = CachingAllocatorConfig::max_split_size();
    context_recorder_.store(nullptr);
  }

  void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      bool alloc_trace_record_context) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    record_history = enabled;
    context_recorder_.store(context_recorder);
    alloc_trace_max_entries_ = std::max(size_t(1), alloc_trace_max_entries);
    alloc_trace_record_context_ = alloc_trace_record_context;
    alloc_trace_next = 0;
    alloc_trace->clear();
  }

  void attachOutOfMemoryObserver(OutOfMemoryObserver observer) {
    oom_observers_.emplace_back(std::move(observer));
  }

  // All public methods (except the above) acquire the allocator mutex.
  // Thus, do not call a public method from another public method.

  Block* malloc(int device, size_t orig_size, cudaStream_t stream) {
    // done outside the lock because we don't know what locks the recorder needs
    // to have...
    CreateContextFn context_recorder = context_recorder_.load();
    std::shared_ptr<Context> context =
        context_recorder ? context_recorder() : nullptr;

    std::unique_lock<std::recursive_mutex> lock(mutex);

    if (C10_LIKELY(captures_underway == 0)) {
      // Processes end-of-life events for outstanding allocations used on
      // multiple streams (checks if their GPU-side uses are complete and
      // recycles their memory if so)
      //
      // Q. Why skip process_events if a capture might be underway?
      // A. process_events involves cudaEventQueries, illegal during CUDA graph
      //    capture.
      //    Dumb simple solution: defer reclaiming these allocations until after
      //    capture. Cross-stream memory use is uncommon, so the deferral's
      //    effect on memory use during capture should be small.
      process_events();
    }
    size_t size = round_size(orig_size);
    auto& pool = get_pool(size, stream);
    const size_t alloc_size = get_allocation_size(size);
    AllocParams params(device, size, stream, &pool, alloc_size, stats);
    params.stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    params.stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] = true;

    // First, try to get a block from the existing pool.
    bool block_found =
        // Search pool
        get_free_block(params)
        // Trigger callbacks and retry search
        || (trigger_free_memory_callbacks(params) && get_free_block(params))
        || get_fused_fragmented_blocks(params, 0);

    // Can't reuse an existing block; try to get a new one.
    if (!block_found) {
      // Do garbage collection if the flag is set.
      if (C10_UNLIKELY(
              set_fraction &&
              CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
        garbage_collect_cached_blocks();
      }
      // Attempt allocate
      block_found = //alloc_block(params, false)
             realloc_block(params, false)
          // Free enough available cached blocks to satisfy alloc and retry
          // alloc.
          || (release_available_cached_blocks(params) &&
              realloc_block(params, false))
          || get_fused_fragmented_blocks(params, 1)
          // Free all non-split cached blocks and retry alloc.
          || (C10_LIKELY(captures_underway == 0) && release_cached_blocks() &&
              realloc_block(params, true))
          || get_fused_fragmented_blocks(params, 2);

      if (record_history && block_found) {
        record_trace(
            TraceEntry::SEGMENT_ALLOC,
            int64_t(params.block->ptr),
            params.block->size,
            params.stream(),
            context);
      }
    }

    if (!block_found) {
      // For any error code other than cudaErrorMemoryAllocation,
      // alloc_block should have thrown an exception already.
      TORCH_INTERNAL_ASSERT(params.err == cudaErrorMemoryAllocation);

      size_t device_free;
      size_t device_total;
      C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
      std::string allowed_info;

      if (set_fraction) {
        allowed_info = format_size(allowed_memory_maximum) + " allowed; ";
      }

      if (record_history) {
        record_trace(
            TraceEntry::OOM,
            device_free,
            params.size(),
            params.stream(),
            std::move(context));
      }
      stats.num_ooms += 1;
      GMLAKE_INFO(" current memory info: device_total: %luMB, device_free: %luMB, request size: %luMB",
                                                                      device_total/(1024*1024), device_free/(1024*1024), size/(1024*1024));
      print_snapshot();

      c10::reportOutOfMemoryToProfiler(
          size,
          stats.allocated_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
              .current,
          stats.reserved_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
              .current,
          c10::Device(c10::DeviceType::CUDA, static_cast<DeviceIndex>(device)));
      for (const auto& obs : oom_observers_) {
        obs(device,
            alloc_size,
            set_fraction ? allowed_memory_maximum : device_total,
            device_free);
      }
      // "total capacity": total global memory on GPU
      // "allowed": memory is allowed to use, which set by fraction.
      // "already allocated": memory allocated by the program using the
      //                      caching allocator
      // "free": free memory as reported by the CUDA API
      // "cached": memory held by the allocator but not used by the program
      //
      // The "allocated" amount  does not include memory allocated outside
      // of the caching allocator, such as memory allocated by other programs
      // or memory held by the driver.
      //
      // The sum of "allocated" + "free" + "cached" may be less than the
      // total capacity due to memory held by the driver and usage by other
      // programs.
      //
      // Note that at this point free_cached_blocks has already returned all
      // possible "cached" memory to the driver. The only remaining "cached"
      // memory is split from a larger block that is partially in-use.
      TORCH_CHECK_WITH(
          OutOfMemoryError,
          false,
          "CUDA out of memory. Tried to allocate ",
          format_size(alloc_size),
          " (GPU ",
          device,
          "; ",
          format_size(device_total),
          " total capacity; ",
          format_size(
              stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)]
                  .current),
          " already allocated; ",
          format_size(device_free),
          " free; ",
          allowed_info,
          format_size(
              stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)]
                  .current),
          " reserved in total by PyTorch)",
          " If reserved memory is >> allocated memory try setting max_split_size_mb to avoid"
          " fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF",
          "");
    }

    TORCH_INTERNAL_ASSERT(
        params.err == cudaSuccess && params.block != nullptr &&
        params.block->ptr != nullptr);
    Block* block = params.block;
    Block* remaining = nullptr;

    static const int vmmDefragment = ([]()->int{
      const char* env = getenv("vmmDefragment");
      if(env) return atoi(env);
      else return 1;
    })();

    const bool already_split = block->is_split();
    if (should_split(block, size)) {
      if(pool.is_small || vmmDefragment <= 0 || (block->vmm_segment && !block->vmm_segment->fused)) {
        
        remaining = block;
          
        block = new Block(device, stream, size, &pool, block->ptr);
        block->prev = remaining->prev;
        if (block->prev) {
          block->prev->next = block;
        }
        block->next = remaining;
          
        remaining->prev = block;
        remaining->ptr = static_cast<char*>(remaining->ptr) + size;
        remaining->size -= size;
          
        if(vmmDefragment > 0 && remaining->vmm_segment) {
              
          auto remaining_segment = remaining->vmm_segment->split(size);
          block->vmm_segment = std::move(remaining->vmm_segment);
          remaining->vmm_segment =  std::move(remaining_segment);
          
              
          size_t offset = 0;
          for(auto& phy_block : block->vmm_segment->phy_blocks) {
            phy_block->mapped_blocks[0].block = block;
            phy_block->mapped_blocks[0].offset = offset;
            phy_block->free = false;
            offset++;
          }
          block->vmm_segment->free_blocks = 0;
          block->vmm_segment->used_blocks = block->vmm_segment->phy_blocks.size();
              
              
          offset = 0;
          for(auto& phy_block : remaining->vmm_segment->phy_blocks) {
            phy_block->mapped_blocks[0].block = remaining;
            phy_block->mapped_blocks[0].offset = offset;
                  
            bool is_prev_free = phy_block->free;
            phy_block->free = true;
                  
            //neglect the the first block, since it is the remaining block
            for(int i=1; i<phy_block->mapped_blocks.size(); i++) {
              Block* other_block = phy_block->mapped_blocks[i].block;
                      
                      
              if(!is_prev_free) {
                other_block->vmm_segment->free_blocks++;
                        
                if(other_block->vmm_segment->fused) {
                  if(other_block->vmm_segment->free_blocks == other_block->vmm_segment->phy_blocks.size()) {
                    if(other_block->stream == block->stream &&
                      fragmented_free_fused_blocks[other_block->stream].blocks.count(other_block)) {
                      fragmented_free_fused_blocks[other_block->stream].erase(other_block);
                                      
                      free_fused_blocks.blocks.insert(other_block);
                      free_fused_blocks_in_release_order[other_block->stream].insert(other_block);
                    }
                  }
                }           
              }            
            }
                  
            offset++;
          }
          remaining->vmm_segment->free_blocks = remaining->vmm_segment->phy_blocks.size();
          remaining->vmm_segment->used_blocks = 0;
        }
          
        bool inserted = pool.blocks.insert(remaining).second;
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);
          
        if (context) {
          trimHistoryBefore(remaining, (char*)block->ptr + size);
        }
          
        if (already_split) {
          // An already-split inactive block is being shrunk by size bytes.
          update_stat_array(
            stats.inactive_split_bytes, -block->size, params.stat_types);
        } else {
          // A new split inactive block is being created from a previously unsplit
          // block, size remaining->size bytes.
          for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
            update_stat(stats.inactive_split_bytes[stat_type], remaining->size);
            update_stat(stats.inactive_split[stat_type], 1);
          });
        }
      }
      else if(vmmDefragment > 0 && block->vmm_segment) {
        size_t keep_blocks = size/kGranularity;
          
        if(block->vmm_segment->used_blocks > keep_blocks) {
          block->vmm_segment->free_blocks = block->vmm_segment->used_blocks - keep_blocks;
          block->vmm_segment->used_blocks = keep_blocks;
              
          for(size_t i=0; i<keep_blocks; i++) {
            if(block->vmm_segment->phy_blocks[i]->free) {
              GMLAKE_INFO(" warning for malloc fused blocks has free phy_block, something wrong happended");
              exit(-1);
            }
          }
              
          std::unordered_set<Block*> blocks2split;
          for(size_t i = keep_blocks; i < block->vmm_segment->phy_blocks.size(); i++) {
            auto& phy_block = block->vmm_segment->phy_blocks[i];
                  
            bool is_prev_free = phy_block->free;
            phy_block->free = true;
                  
            for(auto& block_segment : phy_block->mapped_blocks) {
              Block* other_block = block_segment.block;
              
              if(other_block == block) continue;
                      
              if(!is_prev_free) {
                other_block->vmm_segment->free_blocks++;
                if(other_block->vmm_segment->fused) {
                  if(other_block->vmm_segment->free_blocks == other_block->vmm_segment->phy_blocks.size()) {
                    if(other_block->stream == block->stream &&
                      fragmented_free_fused_blocks[other_block->stream].blocks.count(other_block)) {
                      fragmented_free_fused_blocks[other_block->stream].erase(other_block);
                                      
                      free_fused_blocks.blocks.insert(other_block);
                      free_fused_blocks_in_release_order[other_block->stream].insert(other_block);
                    }
                  }
                } else {
                  if(other_block->vmm_segment->free_blocks == other_block->vmm_segment->phy_blocks.size()) {
                    large_blocks.blocks.insert(other_block);
                                  
                    blocks2split.erase(other_block);
                    
                    if(other_block->is_split()) {
                      update_stat_array(stats.inactive_split, 1, params.stat_types);
                      update_stat_array(stats.inactive_split_bytes, other_block->size, params.stat_types);
                    }
                  } else {
                    if(blocks2split.count(other_block) == 0) {
                      blocks2split.insert(other_block);
                    }
                  }
                } 
              }
            }
          }
    
          for(auto& block2split : blocks2split) {
            if(block2split->vmm_segment->fused || 
               block2split->vmm_segment->free_blocks == 0 || 
               block2split->vmm_segment->free_blocks == block2split->vmm_segment->phy_blocks.size()) continue;
                  
                  
            if(active_blocks.count(block2split)) {
              block2split->allocated = false;
              active_blocks.erase(block2split);
                    
              update_stat_array(stats.active, -1, params.stat_types);
              update_stat_array(stats.active_bytes, -block2split->size, params.stat_types);
            }
                  
                  
            bool block_free = block2split->vmm_segment->phy_blocks[0]->free;          
            size_t last_offset = 0;
            Block* prev_block = block2split->prev;
                  
            auto phy_blocks = block2split->vmm_segment->phy_blocks;
            auto vmm_segment = std::move(block2split->vmm_segment);
                  
            for(size_t i=1; i<=phy_blocks.size(); i++) {
                      
              if( i == phy_blocks.size() || block_free != phy_blocks[i]->free ) {
                size_t block_size = (i - last_offset)*kGranularity;
                          
                char* block_ptr = (char*)block2split->ptr + last_offset*kGranularity;
                Block* split_block = new Block(device, stream, block_size, &pool, block_ptr);
                          
                          
                split_block->prev = prev_block;
                if(prev_block) prev_block->next = split_block;
            
                split_block->self_last_event = block2split->self_last_event;
                          
                          
                if(i < phy_blocks.size()) {
                  auto remaining_segment = vmm_segment->split(block_size);
                  split_block->vmm_segment = std::move(vmm_segment);
                  vmm_segment = std::move(remaining_segment);
                } else {
                  split_block->vmm_segment = std::move(vmm_segment);
                }
                          
                          
                size_t offset = 0;
                for(auto& phy_block : split_block->vmm_segment->phy_blocks) {
                  phy_block->mapped_blocks[0].block = split_block;
                  phy_block->mapped_blocks[0].offset = offset;
                  offset++;
                }
                          
            
                if(block_free) {
                  split_block->vmm_segment->free_blocks = split_block->vmm_segment->phy_blocks.size();
                  split_block->vmm_segment->used_blocks = 0;
              
                  large_blocks.blocks.insert(split_block);
              
                  update_stat_array(stats.inactive_split, 1, params.stat_types);
                  update_stat_array(stats.inactive_split_bytes, split_block->size, params.stat_types);
                } else {
                  split_block->vmm_segment->free_blocks = 0;
                  split_block->vmm_segment->used_blocks = 0;
                              
                  split_block->allocated = true;
                  active_blocks.insert(split_block);
                              
                              
                  update_stat_array(stats.active, 1, params.stat_types);
                  update_stat_array(stats.active_bytes, split_block->size, params.stat_types);
                }
                   

                if(i < phy_blocks.size()) {
                  block_free = phy_blocks[i]->free;
                }
                last_offset = i;
                prev_block = split_block;
              }
            }
                  
                  
            if(prev_block) {
              prev_block->next = block2split->next;
            }
                  
            if(block2split->next) {
              block2split->next->prev = prev_block;
            }
                  
                  
            delete block2split;
          }
        }
      }


      if (record_history) {
        trimHistoryBefore(remaining, (char*)block->ptr + size);
      }

    } else if (already_split) {
      // An already-split block is becoming active
      for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
        update_stat(
            stats.inactive_split_bytes[stat_type],
            -static_cast<std::int64_t>(block->size));
        update_stat(stats.inactive_split[stat_type], -1);
      });
    }

    block->allocated = true;
    block->requested_size = orig_size;
    block->actual_size = size;
    if (record_history) {
      trimHistoryBefore(block, (char*)block->ptr + size);
      block->history = std::make_unique<HistoryChain>(HistoryChain{
          History{block->ptr, orig_size, std::move(context)},
          std::move(block->history)});
      if (!block->history_last) {
        block->history_last = block->history.get();
      }
      record_trace(
          TraceEntry::ALLOC,
          int64_t(block->ptr),
          orig_size,
          block->stream,
          block->history->h.context);
    }

    bool inserted = false;
    if(block->vmm_segment && block->vmm_segment->fused) {
        active_fused_blocks.insert(block);
    } else {
        inserted = active_blocks.insert(block).second;
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);
    }

    for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
      update_stat(stats.allocation[stat_type], 1);
      update_stat(
          stats.allocated_bytes[stat_type],
          static_cast<std::int64_t>(block->actual_size));
      update_stat(
          stats.requested_bytes[stat_type],
          static_cast<std::int64_t>(block->requested_size));
          if (inserted)
          {
              update_stat(stats.active[stat_type], 1);
              update_stat(stats.active_bytes[stat_type], block->size);
          }
    });
    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_allocations, 1);

    c10::reportMemoryUsageToProfiler(
        block->ptr,
        block->size,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::CUDA, device));

    return block;
  }

  void free(Block* block) {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    block->allocated = false;

    // following logic might modifying underlaying Block, causing the size
    // changed. We store ahead for reporting
    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;

    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*(block->pool)))] =
        true;
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.allocation[stat_type], -1);
      update_stat(
          stats.allocated_bytes[stat_type],
          -static_cast<std::int64_t>(block->actual_size));
    });
    if (block->history) {
      record_trace(
          TraceEntry::FREE_REQUESTED,
          int64_t(block->ptr),
          block->history->h.real_size,
          block->stream,
          block->history->h.context);
    }
    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_allocations, -1);

    if (!block->stream_uses.empty()) {
      if (C10_UNLIKELY(captures_underway)) {
        // It's forbidden to cudaEventQuery an event recorded during CUDA graph
        // capture. We conservatively defer recording end-of-life events until
        // the next call to process_events() (which won't happen until no
        // captures are underway)
        needs_events_deferred_until_no_capture.push_back(block);
      } else {
        insert_events(block);
      }
    } else {
      // free_block(block);
      insert_free_event_into_alloc_stream(block);   
      update_block(block);
    }

    c10::reportMemoryUsageToProfiler(
        orig_block_ptr,
        -orig_block_size,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::CUDA, block->device));
  }

  void update_block(Block* block) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    bool flag = false;
      
    std::unordered_set<Block*> blocks2free;
    if(block->vmm_segment) {
          
      for(size_t i=0; i<block->vmm_segment->phy_blocks.size(); i++) {
              
        if(i < block->vmm_segment->used_blocks) {
          auto& phy_block = block->vmm_segment->phy_blocks[i];
              
          bool is_prev_free = phy_block->free;
          if(!is_prev_free) {
            block->vmm_segment->free_blocks++;
            phy_block->free = true;
          } else {
            GMLAKE_INFO(" warning used blocks is free");
            exit(-1);
          }
                  
          for(auto& block_segment : phy_block->mapped_blocks) {
            Block* other_block = block_segment.block;
                      
            if(other_block == block) continue;
                      
            if(!is_prev_free) {
              other_block->vmm_segment->free_blocks++;
                          
              if(other_block->vmm_segment->fused) {
                if(other_block->vmm_segment->free_blocks == other_block->vmm_segment->phy_blocks.size()) {
                  if(other_block->stream == block->stream &&
                    fragmented_free_fused_blocks[other_block->stream].blocks.count(other_block)) {
                    fragmented_free_fused_blocks[other_block->stream].erase(other_block);
                                      
                    free_fused_blocks.blocks.insert(other_block);
                    free_fused_blocks_in_release_order[other_block->stream].insert(other_block);
                  }
                }
              } else {
                if(!other_block->self_last_event ||
                    other_block->self_last_event->event_id < block->self_last_event->event_id) {
                    other_block->self_last_event = block->self_last_event;
                }
                              
                if(other_block->vmm_segment->free_blocks == other_block->vmm_segment->phy_blocks.size()) {
                  blocks2free.insert(other_block);
                }
              }
                          
            }
          }
        }
      }
          
      block->vmm_segment->used_blocks = 0;
          
    }
      
      
    if(block->vmm_segment && block->vmm_segment->fused) {
      if(active_fused_blocks_to_gc.count(block) == 0) {
        if(block->vmm_segment->free_blocks == block->vmm_segment->phy_blocks.size()) {
          if(fragmented_free_fused_blocks[block->stream].blocks.count(block)) {
            fragmented_free_fused_blocks[block->stream].erase(block);
          }
                  
          free_fused_blocks.blocks.insert(block);
          free_fused_blocks_in_release_order[block->stream].insert(block);
        } else {
          fragmented_free_fused_blocks[block->stream].insert(block);
        }
      }

          
      if(active_fused_blocks.count(block)) {
        block->allocated = false;
        active_fused_blocks.erase(block);
        flag = true;
        size_t requested_size = block->requested_size;
        StatTypes stat_types = {false};
        stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
        stat_types[static_cast<size_t>(get_stat_type_for_pool(large_blocks))] = true;
        for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
          update_stat(
            stats.requested_bytes[stat_type],
            -static_cast<std::int64_t>(requested_size));
        });
              
        if(active_fused_blocks_to_gc.count(block)) {
          for(auto& phy_block : block->vmm_segment->phy_blocks) {
            int i = 0;
            for(int j = 0; j < phy_block->mapped_blocks.size(); j++) {
              if(phy_block->mapped_blocks[j].block != block) {
                if(i != j) {
                  phy_block->mapped_blocks[i] = phy_block->mapped_blocks[j];
                }
                              
                i++;
              }
            }
            phy_block->mapped_blocks.resize(i);
          }
                  
          active_fused_blocks_to_gc.erase(block);
          delete block;
        }
      }
    } else {
      free_block(block, flag);
    }
      
      
    for(auto& block2free : blocks2free) {
          
      block2free->allocated = false;
      free_block(block2free, flag);
    }
      
  }

  void* getBaseAllocation(Block* block, size_t* outSize) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    while (block->prev) {
      block = block->prev;
    }
    void* basePtr = block->ptr;
    if (outSize) {
      size_t size = 0;
      while (block) {
        size += block->size;
        block = block->next;
      }
      *outSize = size;
    }
    return basePtr;
  }

  void recordStream(Block* block, cuda::CUDAStream stream) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (stream.stream() == block->stream) {
      // ignore uses on the allocation stream, since those don't require any
      // special synchronization
      return;
    }
    block->stream_uses.insert(stream);
  }

  /** set memory fraction to limit maximum allocated memory **/
  void setMemoryFraction(double fraction) {
    size_t device_free;
    size_t device_total;
    C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
    allowed_memory_maximum = static_cast<size_t>(fraction * device_total);
    set_fraction = true;
  }

  /** returns cached blocks to the system allocator **/
  void emptyCache() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    release_cached_blocks();

    size_t garbage_size = garbage_collect_fused_blocks(2, 0);
    total_fuse_size -= garbage_size;
	
	  GMLAKE_INFO(" garbage_collect_fused_blocks() return %luMB garbage memory", garbage_size/(1024*1024));
  }

  /** Retrieves size of largest unused block held by the memory cache **/
  void cacheInfo(size_t* largest) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (*largest ==
        0) { // make an initial guess if a zero *largest is passed in
      size_t tmp_bytes;
      C10_CUDA_CHECK(cudaMemGetInfo(
          largest, // Use free memory as an optimistic initial guess of *largest
          &tmp_bytes));
    }
    cache_info_aux(large_blocks, largest);
    cache_info_aux(small_blocks, largest);
    for (const auto& gp : graph_pools) {
      cache_info_aux(gp.second->large_blocks, largest);
      cache_info_aux(gp.second->small_blocks, largest);
    }
  }

  /** Returns a copy of the memory allocator stats **/
  DeviceStats getStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats;
  }

  /** Resets the historical accumulation stats for the device **/
  void resetAccumulatedStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      reset_accumulated_stat(stats.allocation[statType]);
      reset_accumulated_stat(stats.segment[statType]);
      reset_accumulated_stat(stats.active[statType]);
      reset_accumulated_stat(stats.inactive_split[statType]);
      reset_accumulated_stat(stats.allocated_bytes[statType]);
      reset_accumulated_stat(stats.reserved_bytes[statType]);
      reset_accumulated_stat(stats.active_bytes[statType]);
      reset_accumulated_stat(stats.inactive_split_bytes[statType]);
      reset_accumulated_stat(stats.requested_bytes[statType]);
    }

    stats.num_alloc_retries = 0;
    stats.num_ooms = 0;
    reset_accumulated_stat(stats.oversize_allocations);
    reset_accumulated_stat(stats.oversize_segments);
  }

  /** Resets the historical peak stats for the device **/
  void resetPeakStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      reset_peak_stat(stats.allocation[statType]);
      reset_peak_stat(stats.segment[statType]);
      reset_peak_stat(stats.active[statType]);
      reset_peak_stat(stats.inactive_split[statType]);
      reset_peak_stat(stats.allocated_bytes[statType]);
      reset_peak_stat(stats.reserved_bytes[statType]);
      reset_peak_stat(stats.active_bytes[statType]);
      reset_peak_stat(stats.inactive_split_bytes[statType]);
      reset_peak_stat(stats.requested_bytes[statType]);
    }
    reset_peak_stat(stats.oversize_allocations);
    reset_peak_stat(stats.oversize_segments);
  }

  /** Dump a complete snapshot of the memory held by the allocator. Potentially
   * VERY expensive. **/
  std::vector<SegmentInfo> snapshot() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    size_t total_active = 0;
    std::vector<SegmentInfo> result;
    const auto all_blocks = get_all_blocks();
    for (const Block* const head_block : all_blocks) {
      if (head_block->prev != nullptr) {
        continue;
      }
      result.emplace_back();
      SegmentInfo& segment_info = result.back();
      segment_info.device = head_block->device;
      segment_info.address = reinterpret_cast<int64_t>(head_block->ptr);
      segment_info.stream = head_block->stream;
      segment_info.is_large = (!head_block->pool->is_small);

      const Block* block = head_block;
      while (block != nullptr) {
        segment_info.blocks.emplace_back();
        BlockInfo& block_info = segment_info.blocks.back();

        block_info.size = block->size;
        block_info.requested_size = block->requested_size;
        block_info.allocated = block->allocated;
        block_info.active = block->allocated || (block->event_count > 0) ||
            !block->stream_uses.empty();

        segment_info.total_size += block_info.size;
        if (block_info.allocated) {
          segment_info.allocated_size += block_info.size;
        }
        if (block_info.active) {
          segment_info.active_size += block_info.size;
          segment_info.requested_size += block_info.requested_size;
        }
        HistoryChain* h = block->history.get();
        while (h) {
          block_info.history.push_back(h->h);
          h = h->next.get();
        }
        block = block->next;
      }
      total_active += segment_info.active_size;
    }

    std::sort(
        result.begin(),
        result.end(),
        [](const SegmentInfo& a, const SegmentInfo& b) {
          return a.address < b.address;
        });

    if (record_history) {
      record_trace(TraceEntry::SNAPSHOT, 0, total_active, 0, nullptr);
    }
    return result;
  }

  std::vector<TraceEntry> trace() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    std::vector<TraceEntry> result;
    result.reserve(alloc_trace->size());
    result.insert(
        result.end(),
        alloc_trace->begin() + alloc_trace_next,
        alloc_trace->end());
    result.insert(
        result.end(),
        alloc_trace->begin(),
        alloc_trace->begin() + alloc_trace_next);
    return result;
  }

  void print_snapshot()
  {
    auto memory_snapshot = snapshot();
      
    for(auto& segment_info : memory_snapshot) {
      if(segment_info.is_large) {
        GMLAKE_INFO(" segment: %p, size: %luMB", (void*)segment_info.address, segment_info.total_size/(1024*1024));
                    
        for(auto& block_info : segment_info.blocks) {
          GMLAKE_INFO(" %s %s block, size: %luMB", 
                     (block_info.allocated? "allocated" : "unallocated"), 
                     (block_info.active? "active" : "inactive"),
                     block_info.size/(1024*1024) );
        }
      }
    }
  }

  // This function takes the size and number of divisions argument and rounds
  // up the size argument for the nearest power-of-2 division.
  // For example, if we need to round-up 1200 and number of divisions is 4,
  // the size 1200 lies between 1024 and 2048 and if we do 4 divisions between
  // them, the values are 1024, 1280, 1536, and 1792. So the function will
  // return 1280 as the nearest ceiling of power-2 divison.
  static size_t roundup_power2_next_division(size_t size, size_t divisions) {
    if (C10_UNLIKELY(size <= 4 || divisions <= 1)) {
      return size;
    }
    if (llvm::isPowerOf2_64(size)) {
      return size;
    }

    // divide the space between these 2's power into equal divisions
    // If division is zero, return the power-of-2 ceiling.
    size_t power2_floor = llvm::PowerOf2Floor(size);
    size_t power2_divison =
        power2_floor >> (63 - llvm::countLeadingZeros(divisions));
    if (C10_UNLIKELY(power2_divison == 0)) {
      return (power2_floor << 1);
    }
    size_t round_size_floor = size & (~(power2_divison - 1));
    return (round_size_floor == size) ? size
                                      : round_size_floor + power2_divison;
  }

  static size_t round_size(size_t size) {
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      auto divisions = CachingAllocatorConfig::roundup_power2_divisions(size);
      if (divisions > 0 && size > (kMinBlockSize * divisions)) {
        return roundup_power2_next_division(size, divisions);
      } else {
        // return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
        size_t block_round_size = kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
        if(block_round_size > kSmallSize) //if block will alloc from large_blocks, round to 2M
        {
            block_round_size = kGranularity * ((size + kGranularity - 1) / kGranularity);
        }
        return block_round_size;
      }
    }
  }

  // See Note [Interaction with CUDA graph capture]

  // Called by CUDAGraph::capture_begin
  void notifyCaptureBegin(CaptureId_t graph_id, MempoolId_t mempool_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    captures_underway++;
    auto it = graph_pools.find(mempool_id);
    if (it == graph_pools.end()) {
      // mempool_id does not reference an existing pool. Make a new pool for
      // this capture.
      graph_pools.emplace(mempool_id, std::make_unique<PrivatePool>());
    } else {
      // mempool_id references an existing pool, which the current capture will
      // share. Check this pool is live (at least one other capture already
      // references it).
      TORCH_INTERNAL_ASSERT(it->second->use_count > 0);
      it->second->use_count++;
    }
    // Maps this graph_id to mempool_id and makes sure this graph_id wasn't
    // somehow assigned a mempool_id already. Keeps essential effect (insert)
    // out of macro.
    bool inserted = capture_to_pool_map.insert({graph_id, mempool_id}).second;
    TORCH_INTERNAL_ASSERT(inserted);
  }

  // Called by CUDAGraph::capture_end
  void notifyCaptureAboutToEnd(CaptureId_t graph_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    captures_underway--;
    auto it = capture_to_pool_map.find(graph_id);
    TORCH_INTERNAL_ASSERT(it != capture_to_pool_map.end());
    capture_to_pool_map.erase(it);
  }

  // Called by CUDAGraph::reset
  void notifyCaptureDestroy(MempoolId_t mempool_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // The instantiated cudaGraphExec_t has been destroyed. We can't blindly
    // delete and cudaFree the mempool its capture used, because
    //  1. other graph(s) might share the same pool
    //  2. the user might still hold references to output tensors allocated
    //  during capture.
    // To handle 1 and 2, we track the number of graphs using this particular
    // mempool. When the count reaches 0, we tell free_cached_blocks it may now
    // cudaFree blocks from this graph's pool when it discovers they're unused
    // (unsplit).
    auto it = graph_pools.find(mempool_id);
    TORCH_INTERNAL_ASSERT(it != graph_pools.end());
    auto uc = --(it->second->use_count);
    TORCH_INTERNAL_ASSERT(uc >= 0);
    if (uc == 0) {
      // Allows free_cached_blocks to begin cudaFreeing this pool's memory,
      // and makes sure this pool wasn't somehow made freeable already.
      bool inserted =
          graph_pools_freeable.insert({mempool_id, it->second.get()}).second;
      TORCH_INTERNAL_ASSERT(inserted);
    }
  }

 private:
  // All private methods do not acquire the allocator mutex.

  std::vector<const Block*> get_all_blocks() const {
    std::vector<const Block*> blocks;
    blocks.insert(
        blocks.end(), small_blocks.blocks.begin(), small_blocks.blocks.end());
    blocks.insert(
        blocks.end(), large_blocks.blocks.begin(), large_blocks.blocks.end());
    for (const auto& gp : graph_pools) {
      blocks.insert(
          blocks.end(),
          gp.second->small_blocks.blocks.begin(),
          gp.second->small_blocks.blocks.end());
      blocks.insert(
          blocks.end(),
          gp.second->large_blocks.blocks.begin(),
          gp.second->large_blocks.blocks.end());
    }
    blocks.insert(blocks.end(), active_blocks.begin(), active_blocks.end());
    return blocks;
  }

  /** moves a block into a pool of cached free blocks */
  void free_block(Block* block, bool flag) {
    TORCH_INTERNAL_ASSERT(
        !block->allocated && block->event_count == 0 &&
        block->stream_uses.empty());
    if (block->history) {
      record_trace(
          TraceEntry::FREE_COMPLETED,
          int64_t(block->ptr),
          block->history->h.real_size,
          block->stream,
          block->history->h.context);
    }
    static const int vmmDefragment = ([]()->int{
        const char* env = getenv("vmmDefragment");
        if(env) return atoi(env);
        else return 1;
    })();

    size_t original_block_size = block->size;
    size_t requested_size = block->requested_size;

    auto& pool = *block->pool;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    const std::array<Block*, 2> merge_candidates = {block->prev, block->next};
    for (Block* merge_candidate : merge_candidates) {
      const int64_t subsumed_size =
          try_merge_blocks(block, merge_candidate, pool);
      if (subsumed_size > 0) {
        net_change_inactive_split_blocks -= 1;
        net_change_inactive_split_size -= subsumed_size;
      }
    }

    active_blocks.erase(block);
    // Makes sure the Block* isn't already present in the pool we're freeing it
    // back into.
    bool inserted = pool.blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT(inserted);

    if(vmmDefragment > 0 && block->vmm_segment/*!pool.is_small*/) {
      block->vmm_segment->free_blocks = block->vmm_segment->phy_blocks.size();
      block->vmm_segment->used_blocks = 0;

    }

    if (block->is_split()) {
      net_change_inactive_split_blocks += 1;
      net_change_inactive_split_size += block->size;
    }

    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] = true;
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(
          stats.inactive_split[stat_type], net_change_inactive_split_blocks);
      update_stat(
          stats.inactive_split_bytes[stat_type],
          net_change_inactive_split_size);
      update_stat(stats.active[stat_type], -1);
      update_stat(
          stats.active_bytes[stat_type],
          -static_cast<std::int64_t>(original_block_size));
      if (!flag) {
      update_stat(
          stats.requested_bytes[stat_type],
          -static_cast<std::int64_t>(requested_size));
      }
    });
  }

  /** combine previously split blocks. returns the size of the subsumed block,
   * or 0 on failure. */
  size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    if (!src || src->allocated || src->event_count > 0 ||
        !src->stream_uses.empty()) {
      return 0;
    }

    AT_ASSERT(dst->is_split() && src->is_split());

    if (dst->prev == src) { // [src dst]
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
      if (!dst->history) {
        dst->history = std::move(src->history);
        dst->history_last = src->history_last;
      } else if (src->history) {
        src->history_last->next = std::move(dst->history);
        dst->history = std::move(src->history);
      }
      src->history_last = nullptr;
    } else { // [dest src]
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }

      if (!dst->history) {
        dst->history = std::move(src->history);
        dst->history_last = src->history_last;
      } else if (src->history) {
        dst->history_last->next = std::move(src->history);
        dst->history_last = src->history_last;
      }
      src->history_last = nullptr;
    }

    std::shared_ptr<BlockEvent> current_self_last_event = src->self_last_event;
    if(!current_self_last_event || (dst->self_last_event && dst->self_last_event->event_id > current_self_last_event->event_id)) {
      current_self_last_event = dst->self_last_event;
    }

    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    dst->self_last_event = current_self_last_event;
    auto erased = pool.blocks.erase(src);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(erased == 1);
    static const int vmmDefragment = ([]()->int{
        const char* env = getenv("vmmDefragment");
        if(env) return atoi(env);
        else return 1;
    })();


    if(vmmDefragment > 0 && dst->vmm_segment) {
      bool ret = dst->vmm_segment->remerge(*(src->vmm_segment));
      if(!ret) {
        GMLAKE_INFO(" merge block %p, ptr %p of size %fMB into block %p, ptr %p of size %fMB failed", 
                    src, src->ptr, src->size/(1024.f*1024.f), dst, dst->ptr, dst->size/(1024.f*1024.f));
      }
      
      size_t offset = 0;
      for(auto& phy_block : dst->vmm_segment->phy_blocks) {
          phy_block->mapped_blocks[0].block = dst;
          phy_block->mapped_blocks[0].offset = offset;
          offset++;
      }
    }

    delete src;

    return subsumed_size;
  }

  BlockPool& get_pool(size_t size, cudaStream_t stream) {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
    // captures_underway is a conservative guess that the current stream may be
    // capturing. It's only > 0 if some thread has begun and not yet ended a
    // capture, so it's usually 0, and we can short-circuit
    // cudaStreamCaptureStatus (which does a TLS lookup).
    if (C10_UNLIKELY(captures_underway)) {
      CaptureId_t id;
      cudaStreamCaptureStatus status;
      C10_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &id));
      if (status != cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) {
        TORCH_INTERNAL_ASSERT(
            status !=
            cudaStreamCaptureStatus::cudaStreamCaptureStatusInvalidated);
        // Retrieves the private pool assigned to this capture.
        auto it0 = capture_to_pool_map.find(id);
        TORCH_INTERNAL_ASSERT(it0 != capture_to_pool_map.end());
        auto it1 = graph_pools.find(it0->second);
        TORCH_INTERNAL_ASSERT(it1 != graph_pools.end());
        if (size <= kSmallSize) {
          return it1->second->small_blocks;
        } else {
          return it1->second->large_blocks;
        }
      }
    }
#endif
    if (size <= kSmallSize) {
      return small_blocks;
    } else {
      return large_blocks;
    }
  }

  StatType get_stat_type_for_pool(const BlockPool& pool) {
    return pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL;
  }

  bool should_split(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->is_small) {
      return remaining >= kMinBlockSize;
    } else {
      return (size < CachingAllocatorConfig::max_split_size()) &&
             (remaining >= kGranularity);
    }
  }

  static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
      return kLargeBuffer;
    } else {
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }

  bool get_free_block(AllocParams& p) {

    static const int vmmDefragment = ([]()->int{
        const char* env = getenv("vmmDefragment");
        if(env) return atoi(env);
        else return 1;
    })();

    static const double reuseLimit = ([]()->double{
        const char* env = getenv("reuseLimit");
        if(env) return atof(env);
        else return 10.0f;
    })();

    
    
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;  

    BlockPool& pool = *p.pool;

    if (C10_UNLIKELY(
            set_fraction &&
            CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
      // Track block reuse interval only when garbage collection is enabled.
      for (auto& b : pool.blocks) {
        ++b->gc_count;
      }
    }
    auto it = pool.blocks.lower_bound(&p.search_key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
      if(vmmDefragment > 0 && !pool.is_small) {
        auto block_it = free_fused_blocks.blocks.lower_bound(&p.search_key);
        if (block_it == free_fused_blocks.blocks.end() 
            || (*block_it)->stream != p.stream() 
            || (*block_it)->size > (p.search_key.size*reuseLimit))
        {
          return false;
        }
                              
            
        p.block = *block_it;
            
            
        size_t keep_blocks = p.search_key.size/kGranularity;
           
        std::unordered_set<Block*> blocks2split;
        for(size_t i=0; i < keep_blocks; i++) {
          auto& phy_block = p.block->vmm_segment->phy_blocks[i];
                
          if(!phy_block->free) {
            GMLAKE_INFO(" warning for fused blocks not free, something wrong happended");
            exit(-1);
          }
                
          phy_block->free = false;

          for(auto& block_segment : phy_block->mapped_blocks) {
            Block* other_block = block_segment.block;
                    
            if(other_block == p.block) continue;
                    
            if(other_block->vmm_segment->fused) {
              if(other_block->vmm_segment->free_blocks == other_block->vmm_segment->phy_blocks.size() && 
                free_fused_blocks.blocks.count(other_block)) {
                  free_fused_blocks.blocks.erase(other_block);
                  free_fused_blocks_in_release_order[other_block->stream].erase(other_block);

                            
                  fragmented_free_fused_blocks[other_block->stream].insert(other_block);
              } else if(active_fused_blocks.count(other_block) == 0) {
                if(fragmented_free_fused_blocks[other_block->stream].blocks.count(other_block) == 0) {
                  fragmented_free_fused_blocks[other_block->stream].insert(other_block);
                }
              }
                        
                        
              other_block->vmm_segment->free_blocks--;
            } else {
              if(other_block->vmm_segment->free_blocks == other_block->vmm_segment->phy_blocks.size()) {
                if(large_blocks.blocks.count(other_block)) {
                  large_blocks.blocks.erase(other_block);
                                 
                  blocks2split.insert(other_block);
               
                  if(other_block->is_split()) {
                    net_change_inactive_split_blocks -= 1;
                    net_change_inactive_split_size -= other_block->size;
                  }
                }
              }
                        
                        
              other_block->vmm_segment->free_blocks--;
                        
                        
              if(other_block->vmm_segment->free_blocks == 0) {
                blocks2split.erase(other_block);
                            
                other_block->allocated = true;
                active_blocks.insert(other_block);
                            
                            
                update_stat_array(stats.active, 1, p.stat_types);
                update_stat_array(stats.active_bytes, other_block->size, p.stat_types);
              }
            }
          }
        }
            
            
        for(auto& block2split : blocks2split) {      
          if(block2split->vmm_segment->fused || 
            block2split->vmm_segment->free_blocks == 0 || 
            block2split->vmm_segment->free_blocks == block2split->vmm_segment->phy_blocks.size()) {
                    continue;
          }
                
                
          bool block_free = block2split->vmm_segment->phy_blocks[0]->free;
          size_t last_offset = 0;
          Block* prev_block = block2split->prev;
                
          auto phy_blocks = block2split->vmm_segment->phy_blocks;
          auto vmm_segment = std::move(block2split->vmm_segment);
                
          for(size_t i=1; i <= phy_blocks.size(); i++) {
                    
            if(i == phy_blocks.size() || block_free != phy_blocks[i]->free) {
              size_t block_size = (i - last_offset)*kGranularity;
                        
              char* block_ptr = (char*)block2split->ptr + last_offset*kGranularity;
              Block* split_block = new Block(p.device(), p.stream(), block_size, p.pool, block_ptr);
                        
                        
              split_block->prev = prev_block;
              if(prev_block) {
                prev_block->next = split_block;
              }
              split_block->self_last_event = block2split->self_last_event;
                        
                        
              if(i < phy_blocks.size()) {
                auto remaining_segment = vmm_segment->split(block_size);
                split_block->vmm_segment = std::move(vmm_segment);
                vmm_segment = std::move(remaining_segment);
              } else {
                split_block->vmm_segment = std::move(vmm_segment);
              }
                        
                        
              size_t offset = 0;
              for(auto& phy_block : split_block->vmm_segment->phy_blocks) {
                phy_block->mapped_blocks[0].block = split_block;
                phy_block->mapped_blocks[0].offset = offset;
                offset++;
              }


              if(block_free) {
                split_block->vmm_segment->free_blocks = split_block->vmm_segment->phy_blocks.size();
                split_block->vmm_segment->used_blocks = 0;
                            
                            
                large_blocks.blocks.insert(split_block);
                            
                            
                net_change_inactive_split_blocks += 1;
                net_change_inactive_split_size += split_block->size;
              } else {
                split_block->vmm_segment->free_blocks = 0;
                split_block->vmm_segment->used_blocks = 0;
                            
                split_block->allocated = true;
                active_blocks.insert(split_block);
                            
                            
                update_stat_array(stats.active, 1, p.stat_types);
                update_stat_array(stats.active_bytes, split_block->size, p.stat_types);
              }
          

              if(i < phy_blocks.size()) {
                block_free = phy_blocks[i]->free;
              }
              last_offset = i;
              prev_block = split_block;
            }
          }
                
                
          if(prev_block) {
            prev_block->next = block2split->next;
          }
                
          if(block2split->next) {
            block2split->next->prev = prev_block;
          }
                
          delete block2split;
        }
            
        p.block->vmm_segment->free_blocks = (p.block->vmm_segment->phy_blocks.size() - keep_blocks);
        p.block->vmm_segment->used_blocks = keep_blocks;

              
        free_fused_blocks.blocks.erase(block_it);
        free_fused_blocks_in_release_order[p.block->stream].erase(p.block);
    
        p.err = cudaSuccess;
    
        update_stat_array(stats.inactive_split, net_change_inactive_split_blocks, p.stat_types);
        update_stat_array(stats.inactive_split_bytes, net_change_inactive_split_size, p.stat_types);
    
        return true;
      }
        
      return false;
    }
    // Do not return an oversized block for a large request
    if ((p.size() < CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= CachingAllocatorConfig::max_split_size()))
      return false;
    // Allow oversized block size to be rounded up but within a limit
    if ((p.size() >= CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= p.size() + kLargeBuffer))
      return false;
    p.block = *it;
    (*it)->gc_count = 0; // Denote this block has been used
    pool.blocks.erase(it);
    if (vmmDefragment > 0 && p.block->vmm_segment) {
      for(size_t i=0; i < p.block->vmm_segment->phy_blocks.size(); i++) {
        auto& phy_block = p.block->vmm_segment->phy_blocks[i];
        if(!phy_block->free) {
          GMLAKE_INFO(" warning for non fused blocks has non free phy_block: %lu, something wrong happended, block %p, block->ptr %p, block->size %fMB, free_blocks %lu, used_blocks %lu, event_id: %lu",
                      i, p.block, p.block->ptr, p.block->size/(1024.f*1024.f), p.block->vmm_segment->free_blocks, p.block->vmm_segment->used_blocks, p.block->self_last_event->event_id);
            
              
          for(auto& block_segment : phy_block->mapped_blocks) {
            Block* other_block = block_segment.block;
                  
            if(other_block == p.block) continue;
              
            GMLAKE_INFO(" warning for non fused blocks has non free phy_block: %lu, something wrong happended, co-ref block %p, block->ptr %p, block->size %fMB, free_blocks %lu, used_blocks %lu, event_id: %lu",
                        i, other_block, other_block->ptr, other_block->size/(1024.f*1024.f), other_block->vmm_segment->free_blocks, other_block->vmm_segment->used_blocks, other_block->self_last_event->event_id);
          }
      
          exit(-1);
        }
          
        phy_block->free = false;
          
        for(auto& block_segment : phy_block->mapped_blocks) {
          Block* other_block = block_segment.block;
              
          if(other_block == p.block) continue;
              
          if(other_block->vmm_segment->fused) {
            if(other_block->vmm_segment->free_blocks == other_block->vmm_segment->phy_blocks.size() && 
              free_fused_blocks.blocks.count(other_block)) {
              free_fused_blocks.blocks.erase(other_block);
              free_fused_blocks_in_release_order[other_block->stream].erase(other_block);

              fragmented_free_fused_blocks[other_block->stream].insert(other_block);
            } else if(active_fused_blocks.count(other_block) == 0) {
              if(fragmented_free_fused_blocks[other_block->stream].blocks.count(other_block) == 0) {
                fragmented_free_fused_blocks[other_block->stream].insert(other_block);
              }
            }
                                
                  
            other_block->vmm_segment->free_blocks--;
          } else {
            GMLAKE_INFO(" warning for non fused blocks has phy_block mapped to other non fused blocks");
            exit(-1);
          }
        }
                        
      }
      p.block->vmm_segment->free_blocks = 0;
      p.block->vmm_segment->used_blocks = p.block->vmm_segment->phy_blocks.size();

      
    }
    
    
    
    p.err = cudaSuccess;
    
    
    update_stat_array(stats.inactive_split, net_change_inactive_split_blocks, p.stat_types);
    update_stat_array(stats.inactive_split_bytes, net_change_inactive_split_size, p.stat_types);
    return true;
  }

  size_t garbage_collect_fused_blocks(int time, size_t require_size = 0) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
      
    size_t garbage_size = 0;
    size_t garbage_blocks = 0;
    for(auto& it : fragmented_free_fused_blocks) {
      for(auto block_it = it.second.blocks.begin(); block_it != it.second.blocks.end();) {
        Block* block = (*block_it);
      
        cudaError_t err = cudaSuccess;
        if(block->self_last_event) {
          err = cudaEventQuery(block->self_last_event->event);
        }
              
        if(err == cudaSuccess) {
          for(auto& phy_block : block->vmm_segment->phy_blocks) {
            int i = 0;
            for(int j = 0; j < phy_block->mapped_blocks.size(); j++) {
              if(phy_block->mapped_blocks[j].block != block) {
                if(i != j) {
                  phy_block->mapped_blocks[i] = phy_block->mapped_blocks[j];
                }
                              
                i++;
              }
            }
            phy_block->mapped_blocks.resize(i);
          }
                  
          garbage_blocks++;
          garbage_size += block->size;
                  
          //free_fused_blocks.blocks.erase(block);
          block_it = it.second.erase(block_it);
                  
                  
          if(!block->vmm_segment.unique()) {
            GMLAKE_INFO(" warning block is not unique, ref_count: %lu, block %p, block->ptr %p, block->size %fMB, phy_blocks %lu, free_blocks %lu, used_blocks %lu, event_id: %lu", 
                        block->vmm_segment.use_count(), block, block->ptr, block->size/(1024.f*1024.f), block->vmm_segment->phy_blocks.size(), block->vmm_segment->free_blocks, block->vmm_segment->used_blocks, block->self_last_event->event_id);
            exit(-1);
          }
                  
                  
          if(block->vmm_segment->vir_blocks[0]->vir_dev_ptr.use_count() != block->vmm_segment->vir_blocks.size()) {
            GMLAKE_INFO(" warning vir_blocks vir_dev_ptr use_count %lu != vir_blocks.size() %lu, block %p, block->ptr %p, block->size %fMB, phy_blocks %lu, free_blocks %lu, used_blocks %lu, event_id: %lu", 
                        block->vmm_segment->vir_blocks[0]->vir_dev_ptr.use_count(), block->vmm_segment->vir_blocks.size(),
                        block, block->ptr, block->size/(1024.f*1024.f), block->vmm_segment->phy_blocks.size(), block->vmm_segment->free_blocks, block->vmm_segment->used_blocks, block->self_last_event->event_id);
            exit(-1);
          }
                  
          {
            //block->vmm_segment.reset();
            auto tmp = std::move(block->vmm_segment);
          }
                  
          delete block;
                  
          if(require_size > 0 && time <= 1 && garbage_size >= (require_size << (2*(time + 1))) ) break;
          
        } else if(err == cudaErrorNotReady) {
                  
          GMLAKE_INFO(" fragmented_free_fused_blocks: block self_last_event NotReady %p, block->ptr %p, block->size %fMB, phy_blocks %lu, free_blocks %lu, used_blocks %lu, event_id: %lu", 
                      block, block->ptr, block->size/(1024.f*1024.f), block->vmm_segment->phy_blocks.size(), block->vmm_segment->free_blocks, block->vmm_segment->used_blocks, block->self_last_event->event_id);
                  
                  
          cudaGetLastError();
          break;
        } else {
          C10_CUDA_CHECK(err);
          cudaGetLastError();
          break;
        }
      }
    }
      
      
    GMLAKE_INFO(" gc from fragmented_free_fused_blocks: blocks %lu, size %fMB", garbage_blocks, garbage_size/(1024.f*1024.f));
      
      
    if(time > 0) {
      for(auto& it : free_fused_blocks_in_release_order) {
        for(auto block_it = it.second.blocks.begin(); block_it != it.second.blocks.end();) {
          Block* block = (*block_it);
      
          cudaError_t err = cudaSuccess;
          if(block->self_last_event) {
            err = cudaEventQuery(block->self_last_event->event);
          }
                
          if(err == cudaSuccess) {
            for(auto& phy_block : block->vmm_segment->phy_blocks) {
              int i = 0;
              for(int j = 0; j < phy_block->mapped_blocks.size(); j++) {
                if(phy_block->mapped_blocks[j].block != block) {
                  if(i != j) {
                    phy_block->mapped_blocks[i] = phy_block->mapped_blocks[j];
                  }
                                
                  i++;
                }
              }
                        
              phy_block->mapped_blocks.resize(i);
            }
                    
            garbage_blocks++;
            garbage_size += block->size;
                    
            free_fused_blocks.blocks.erase(block);
            block_it = it.second.erase(block_it);
                  
                            
            if(!block->vmm_segment.unique()) {
              GMLAKE_INFO(" warning block is not unique, ref_count %lu, block %p, block->ptr %p, block->size %fMB, phy_blocks %lu, free_blocks %lu, used_blocks %lu, event_id: %lu", 
                          block->vmm_segment.use_count(), block, block->ptr, block->size/(1024.f*1024.f), block->vmm_segment->phy_blocks.size(), block->vmm_segment->free_blocks, block->vmm_segment->used_blocks, block->self_last_event->event_id);
              exit(-1);
            }
                    
                    
                  
            if(block->vmm_segment->vir_blocks[0]->vir_dev_ptr.use_count() != block->vmm_segment->vir_blocks.size()) {
              GMLAKE_INFO(" warning vir_blocks vir_dev_ptr use_count %lu != vir_blocks.size() %lu, block %p, block->ptr %p, block->size %fMB, phy_blocks %lu, free_blocks %lu, used_blocks %lu, event_id: %lu", 
                          block->vmm_segment->vir_blocks[0]->vir_dev_ptr.use_count(), block->vmm_segment->vir_blocks.size(),
                          block, block->ptr, block->size/(1024.f*1024.f), block->vmm_segment->phy_blocks.size(), block->vmm_segment->free_blocks, block->vmm_segment->used_blocks, block->self_last_event->event_id);
              exit(-1);
            }
                    
            delete block;
          } else if(err == cudaErrorNotReady) {
            GMLAKE_INFO(" free_fused_blocks_in_release_order: block self_last_event NotReady %p, block->ptr %p, block->size %fMB, phy_blocks %lu, free_blocks %lu, used_blocks %lu, event_id: %lu", 
                        block, block->ptr, block->size/(1024.f*1024.f), block->vmm_segment->phy_blocks.size(), block->vmm_segment->free_blocks, block->vmm_segment->used_blocks, block->self_last_event->event_id);
                    
            cudaGetLastError();
            break;
          } else {
            C10_CUDA_CHECK(err);
            cudaGetLastError();
            break;
          }
        }
      }   
    }
      
      //cudaDeviceSynchronize();
      
    GMLAKE_INFO(" gc from free_fused_blocks_in_release_order: blocks %lu, size %fMB", garbage_blocks, garbage_size/(1024.f*1024.f));

    return garbage_size;
  }

  bool get_fused_fragmented_blocks(AllocParams& p, int time) {
    static const int vmmDefragment = ([]()->int{
        const char* env = getenv("vmmDefragment");
        if(env) return atoi(env);
        else return 1;
    })();
    
    
    static const size_t fragment_limit = ([]()->size_t{
        const char* env = getenv("fragLimit");
        if(env) return (size_t)std::stoll(env);
        else return (size_t)(32*1024*1024);
    })();
    
    
    static const int defragment_level = ([]()->int{
        const char* env = getenv("defragLevel");
        if(env) return (int)std::atoi(env);
        else return (int)1;
    })();
    
    
    static const int auto_gc_limits = ([]()->int{
        const char* env = getenv("autoGC");
        if(env) return (int)std::atoi(env);
        else return (int)80;
    })();
    
    
    if (vmmDefragment <= 0) {
      return false;
    }
    
    
    if(time < defragment_level) {
        return false;
    }
    
    
    if (p.pool->is_small || p.search_key.size < fragment_limit) {
      return false;
    } else {
      Block left_search_key(p.search_key.device, p.search_key.stream, p.search_key.size, p.search_key.pool, p.search_key.ptr);
      Block right_search_key(p.search_key.device, p.search_key.stream, p.search_key.size, p.search_key.pool, p.search_key.ptr);

      left_search_key.size = 0;
      right_search_key.size = std::numeric_limits<size_t>::max();

      auto it_begin = large_blocks.blocks.lower_bound(&left_search_key);
      if (it_begin == large_blocks.blocks.end() || (*it_begin)->stream != p.stream())
        return false;
      
      auto it_end = large_blocks.blocks.lower_bound(&right_search_key);
      if (it_end == large_blocks.blocks.begin() || (*std::prev(it_end))->stream != p.stream())
        return false;
      
      
      if(std::prev(it_end) == it_begin) return false;
      
      size_t fuse_size = 0;
      std::vector<Block*> blocks2fuse;
      
      auto it = it_end;
      while(it != it_begin && fuse_size < p.search_key.size) {
        it = std::prev(it);
        blocks2fuse.push_back((*it));
        fuse_size += (*it)->size;
      }
      
      
      if(fuse_size < p.search_key.size) {
          return false;
      }
            
      
      int64_t net_change_segments = 0;
      int64_t net_change_inactive_split_blocks = 0;
      int64_t net_change_inactive_split_size = 0;
      
      
      std::shared_ptr<BlockEvent> current_self_last_event;
      std::vector<std::shared_ptr<PhyBlock>> phy_blocks2glue;
      int index = 0;  
      for(auto& block : blocks2fuse) {
        for(auto& phy_block : block->vmm_segment->phy_blocks) {
          phy_block->free = false;
          phy_blocks2glue.push_back(phy_block);
        }
        block->vmm_segment->free_blocks = 0;
        block->vmm_segment->used_blocks = 0;

        
        if(!current_self_last_event || 
          (block->self_last_event && block->self_last_event->event_id > current_self_last_event->event_id)) {
          current_self_last_event = block->self_last_event;
        }
        
        large_blocks.blocks.erase(block);
        
        
        if(block->is_split()) {
          net_change_inactive_split_blocks -= 1;
          net_change_inactive_split_size -= block->size;
        }
        
        
        
        block->allocated = true;
        active_blocks.insert(block);
        
        if (index == blocks2fuse.size() - 1 && (fuse_size - p.search_key.size) >= kGranularity) continue;
        update_stat_array(stats.active, 1, p.stat_types);
        update_stat_array(stats.active_bytes, block->size, p.stat_types);
        index++;
      }

      if(fuse_size > p.search_key.size && (fuse_size - p.search_key.size) >= kGranularity) {
        Block* last_block = blocks2fuse.back();
          
          
        last_block->allocated = false;
        if(active_blocks.count(last_block)) {
          active_blocks.erase(last_block);
        }
                      
          
        Block* remaining = last_block;
      
        size_t original_size = remaining->size;
        size_t remain_size = (fuse_size - p.search_key.size);
        size_t keep_size = original_size - remain_size;
  
        last_block = new Block(p.device(), p.stream(), keep_size, p.pool, last_block->ptr);
        last_block->prev = remaining->prev;
        if (last_block->prev) {
            last_block->prev->next = last_block;
        }
        last_block->next = remaining;
        last_block->self_last_event = remaining->self_last_event;
          
        remaining->prev = last_block;
        remaining->ptr = static_cast<char*>(remaining->ptr) + keep_size;
        remaining->size = remain_size;
          
        auto remaining_segment = remaining->vmm_segment->split(keep_size);
        last_block->vmm_segment = std::move(remaining->vmm_segment);
        remaining->vmm_segment =  std::move(remaining_segment);
          
        for(size_t i=0; i<last_block->vmm_segment->phy_blocks.size(); i++) {
          last_block->vmm_segment->phy_blocks[i]->mapped_blocks[0].block = last_block;
          last_block->vmm_segment->phy_blocks[i]->mapped_blocks[0].offset = i;
          last_block->vmm_segment->phy_blocks[i]->free = false;
        }
        last_block->vmm_segment->free_blocks = 0;
        last_block->vmm_segment->used_blocks = 0;
        last_block->allocated = true;
                    
        active_blocks.insert(last_block);

        update_stat_array(stats.active, 1, p.stat_types);
        update_stat_array(stats.active_bytes, last_block->size, p.stat_types);
          
        for(size_t i=0; i<remaining->vmm_segment->phy_blocks.size(); i++) {
          remaining->vmm_segment->phy_blocks[i]->mapped_blocks[0].block = remaining;
          remaining->vmm_segment->phy_blocks[i]->mapped_blocks[0].offset = i;
          remaining->vmm_segment->phy_blocks[i]->free = true;
        }
        remaining->vmm_segment->free_blocks = remaining->vmm_segment->phy_blocks.size();
        remaining->vmm_segment->used_blocks = 0;
        remaining->allocated = false;
          
        large_blocks.blocks.insert(remaining);
          
        fuse_size -= remaining->size;
  
        size_t keep_blocks = p.search_key.size/kGranularity;
        phy_blocks2glue.resize(keep_blocks);
            
        net_change_inactive_split_blocks += 1;
        net_change_inactive_split_size += remaining->size;
      }
      
      static constexpr size_t G=1024*1024*1024;
      
      using Ms = std::chrono::duration<double, std::milli>;
      Ms fuse_time = Ms{0};
      
      std::shared_ptr<VmmSegment> vmm_segment;
      int gc_time = 0;
      do
      {
        auto t0 = std::chrono::steady_clock::now();
          
        vmm_segment = std::make_shared<VmmSegment>(std::move(phy_blocks2glue));
          
        auto t1 = std::chrono::steady_clock::now();
        fuse_time = (t1-t0);
          
        if(vmm_segment->status == CUDA_SUCCESS && vmm_segment->segment_ptr) {
          break;
        } else {
          cudaGetLastError();
              
          phy_blocks2glue = std::move(vmm_segment->phy_blocks);
              
          GMLAKE_INFO(" allocate virtual address for %lu phy_blocks the %dth time failed, try to garbage_collect_fused_blocks", phy_blocks2glue.size(), gc_time);
              
          size_t garbage_size = garbage_collect_fused_blocks(gc_time, p.search_key.size);
          gc_time++;
              
          total_fuse_size -= garbage_size;
              
          cudaGetLastError();
        }
      } while(gc_time < 3);
      
      if(!vmm_segment || vmm_segment->status != CUDA_SUCCESS || !vmm_segment->segment_ptr) {
          return false;
      }
      
      void* block_ptr = vmm_segment->segment_ptr;
      Block* fused_block = new Block(p.device(), p.stream(), fuse_size, p.pool, (char*)block_ptr);

      fused_block->vmm_segment = std::move(vmm_segment);
      fused_block->self_last_event = current_self_last_event;

      for(auto& phy_block : fused_block->vmm_segment->phy_blocks) {
        for(auto& block_segment : phy_block->mapped_blocks) {
          Block* other_block = block_segment.block;
              
          //since the non fused blocks has already been processed, we only need to process fused blocks 
          if(other_block->vmm_segment->fused) {
            if(other_block->vmm_segment->free_blocks == other_block->vmm_segment->phy_blocks.size() && 
              free_fused_blocks.blocks.count(other_block)) {
              free_fused_blocks.blocks.erase(other_block);
              free_fused_blocks_in_release_order[other_block->stream].erase(other_block);
        
              fragmented_free_fused_blocks[other_block->stream].insert(other_block);
            } else if(active_fused_blocks.count(other_block) == 0) {
              if(fragmented_free_fused_blocks[other_block->stream].blocks.count(other_block) == 0) {
                fragmented_free_fused_blocks[other_block->stream].insert(other_block);
              }
            }
                  

            other_block->vmm_segment->free_blocks--;
          }
        }
      }

      size_t offset = 0;
      for(auto& phy_block : fused_block->vmm_segment->phy_blocks) {
        phy_block->mapped_blocks.emplace_back(fused_block, offset);
        offset++;
      }
      fused_block->vmm_segment->free_blocks = 0;
      fused_block->vmm_segment->used_blocks = fused_block->vmm_segment->phy_blocks.size();

      p.block = fused_block;
      p.err = cudaSuccess;

      GMLAKE_INFO(" fused block %p, ptr %p of size %fMB", 
			            fused_block, fused_block->ptr, fused_block->size/(1024.f*1024.f));
      
      net_change_segments += 1;


      update_stat_array(stats.segment, net_change_segments, p.stat_types);
      update_stat_array(stats.inactive_split, net_change_inactive_split_blocks, p.stat_types);
      update_stat_array(stats.inactive_split_bytes, net_change_inactive_split_size, p.stat_types);

      if(fuse_size >= p.search_key.size) {
        total_fuse_size += fuse_size;
        GMLAKE_INFO(" try %d: fuse %lu physical blocks to ptr %p of size %fMB for allocate size %fMB succeeded, takes %fms, total_fuse_size %fMB", 
                   time, fused_block->vmm_segment->phy_blocks.size(), fused_block->vmm_segment->segment_ptr, fuse_size/(1024.f*1024.f), p.search_key.size/(1024.f*1024.f), fuse_time.count(), total_fuse_size/(1024.f*1024.f));
        
        if(total_fuse_size > auto_gc_limits*G) {
            GMLAKE_INFO(" virtual address larger than %luG, do garbage_collect_fused_blocks() ", auto_gc_limits);
            
            size_t garbage_size = garbage_collect_fused_blocks(2, 0);
            
            total_fuse_size -= garbage_size;
        }
      }

      return fuse_size >= p.search_key.size;
    } 
    
    return false;
  }

  bool trigger_free_memory_callbacks(AllocParams& p) {
    bool freed_memory = false;
    for (const auto& name : FreeCudaMemoryCallbacksRegistry()->Keys()) {
      freed_memory |=
          FreeCudaMemoryCallbacksRegistry()->Create(name)->Execute();
    }
    return freed_memory;
  }

  void garbage_collect_cached_blocks() {
    // Free unused cached blocks to reclaim GPU memory.
    // Unlike release_cached_blocks(), this does not enforce synchronization and
    // therefore should be of less overheads.

    size_t gc_threshold = static_cast<size_t>(
        CachingAllocatorConfig::garbage_collection_threshold() *
        allowed_memory_maximum);
    // No need to trigger GC yet
    if (total_allocated_memory <= gc_threshold) {
      return;
    }
    const auto target_size = total_allocated_memory - gc_threshold;
    size_t gc_reclaimed = 0;

    // Calculate the total age of the free-able blocks. We'll use it later to
    // get "avg age" threshold.
    double total_age = 0.0;
    int freeable_block_count = 0;
    for (auto& b : large_blocks.blocks) {
      if (!b->is_split()) {
        total_age += b->gc_count;
        ++freeable_block_count;
      }
    }
    // No free-able blocks?
    if (freeable_block_count == 0) {
      return;
    }

    // Repeat GC until we reach reclaim > target size.
    bool block_freed = true;
    while (gc_reclaimed < target_size && block_freed == true &&
           freeable_block_count > 0) {
      // Free blocks exceeding this age threshold first.
      double age_threshold = total_age / freeable_block_count;
      // Stop iteration if we can no longer free a block.
      block_freed = false;

      // Free blocks of > avg age. Don't stop upon reaching the target_size,
      // we don't want this GC to be triggered frequently.
      auto it = large_blocks.blocks.begin();
      while (it != large_blocks.blocks.end()) {
        Block* block = *it;
        ++it;
        if (!block->is_split() && block->gc_count >= age_threshold) {
          block_freed = true;
          gc_reclaimed += block->size;
          total_age -= block->gc_count; // Decrement the age
          freeable_block_count--; // One less block that can be freed
          release_block(block);
        }
      }
    }
  }

  bool realloc_block(AllocParams& p, bool isRetry) {
    // Defensively checks for preexisting CUDA error state.
    C10_CUDA_CHECK(cudaGetLastError());

    
    static const int vmmDefragment = ([]()->int{
        const char* env = getenv("vmmDefragment");
        if(env) return atoi(env);
        else return 1;
    })();


    static const int reAlloc = ([]()->int{
        const char* env = getenv("reAlloc");
        if(env) return atoi(env);
        else return 0;
    })();



    size_t size = p.alloc_size;
    size_t free_block_size = 0;
    void* ptr;

    if (isRetry) {
      stats.num_alloc_retries += 1;
    }

    std::shared_ptr<VmmSegment> vmm_segment;
    if (set_fraction &&
        total_allocated_memory + size > allowed_memory_maximum) {
      p.err = cudaErrorMemoryAllocation;
      return false;
    } else {
      if(vmmDefragment <= 0 || p.pool->is_small) {
        p.err = cudaMallocMaybeCapturing(&ptr, size);
        if (p.err != cudaSuccess) {
          if (p.err == cudaErrorMemoryAllocation) {
            // If this is the first attempt (!isRetry), we can forgive and clear CUDA's
            //   internal error state.
            // If this is the second attempt (isRetry), malloc's TORCH_CHECK_WITH will take
            //   over to throw a helpful exception. The user can choose to catch the exception,
            //   free some stuff in their script, and attempt their allocation again.
            //   In this case, we can also forgive and clear CUDA's internal error state.
            cudaGetLastError();
          } else {
            // If the error's unrelated to memory allocation, we should throw immediately.
            C10_CUDA_CHECK(p.err);
          }
          return false;
        }
      } else {
        if(reAlloc > 0) {
          //Block left_search_key = p.search_key;
          //Block right_search_key = p.search_key;
                
          Block left_search_key(p.search_key.device, p.search_key.stream, p.search_key.size, p.search_key.pool, p.search_key.ptr);
          Block right_search_key(p.search_key.device, p.search_key.stream, p.search_key.size, p.search_key.pool, p.search_key.ptr);
      
          left_search_key.size = 0;
          right_search_key.size = std::numeric_limits<size_t>::max();
                      
          auto it_begin = large_blocks.blocks.lower_bound(&left_search_key);
          auto it_end = large_blocks.blocks.lower_bound(&right_search_key);
                
          if(it_begin != large_blocks.blocks.end() && (*it_begin)->stream == p.stream() &&
            it_end != large_blocks.blocks.begin() && (*std::prev(it_end))->stream == p.stream()) {
            auto it = it_begin;
            while(it != it_end) {
              free_block_size += (*it)->size;
              it++;
            }
          }
                
                
          size_t request_size = p.search_key.size;
                
          if(free_block_size >= request_size) {
            GMLAKE_INFO(" free_block_size %fMB is larger than allocation size %fMB, something weired happended", 
                       free_block_size/(1024.f*1024.f), size/(1024.f*1024.f));
            return false;
          }
                
          if(free_block_size > 0) {
            request_size -= free_block_size;
            size = get_allocation_size(request_size);
          }
        }
               
        using Ms = std::chrono::duration<double, std::milli>;
        Ms fuse_time = Ms{0};
            
        int gc_time = 0;
        do
        {
          auto t0 = std::chrono::steady_clock::now();
                
          vmm_segment = std::make_shared<VmmSegment>(size/kGranularity, kGranularity, p.device());
                
          auto t1 = std::chrono::steady_clock::now();
          fuse_time = (t1-t0);
                
          if(vmm_segment->status == CUDA_SUCCESS && vmm_segment->segment_ptr) {
            break;
          } else {
            cudaGetLastError();
                            
            size_t device_free;
            size_t device_total;
            cudaMemGetInfo(&device_free, &device_total);
                            
            size_t total_garbage_size = fragmented_free_fused_blocks[p.stream()].pool_size + free_fused_blocks_in_release_order[p.stream()].pool_size;
                  
                    
            if(device_free > size && total_garbage_size >= size) {
              GMLAKE_INFO(" allocate size %luMB memory by vmm the %dth time failed, try to garbage_collect_fused_blocks", size/(1024*1024), gc_time);
                        
              vmm_segment.reset();
              size_t garbage_size = garbage_collect_fused_blocks(gc_time, p.alloc_size);
              total_fuse_size -= garbage_size;
                        
              gc_time++;
                       
              cudaGetLastError();
            } else {
              break;
            }
          }
        } while(gc_time < 3);
            
        if(!vmm_segment || vmm_segment->status != CUDA_SUCCESS || !vmm_segment->segment_ptr) {           
          p.err = cudaErrorMemoryAllocation;
          cudaGetLastError();
          vmm_segment.reset();
                
          GMLAKE_INFO(" allocate size %fMB memory by vmm failed", size/(1024.f*1024.f));
            
          return false;
        }
            
            
        ptr = vmm_segment->segment_ptr;
      }
    }

    if (p.pool->owner_PrivatePool) {
      // The block is for a CUDA graph's PrivatePool.
      p.pool->owner_PrivatePool->cudaMalloc_count++;
    }

    total_allocated_memory += size;
    Block* new_block = new Block(p.device(), p.stream(), size, p.pool, (char*)ptr);
    new_block->vmm_segment = std::move(vmm_segment);
    
    for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
      update_stat(stats.segment[stat_type], 1);
      update_stat(stats.reserved_bytes[stat_type], size);
    });
    if (size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_segments, 1);

    // p.block came from new, not cudaMalloc. It should not be nullptr here.
    //TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);
    TORCH_INTERNAL_ASSERT(new_block != nullptr && new_block->ptr != nullptr);
    
    
    if(new_block->vmm_segment) {
      if(new_block->size < p.search_key.size) {
        for(size_t i = 0; i < new_block->vmm_segment->phy_blocks.size(); i++) {
          new_block->vmm_segment->phy_blocks[i]->mapped_blocks.emplace_back(new_block, i);
          new_block->vmm_segment->phy_blocks[i]->free = true;
        }
            
        new_block->vmm_segment->free_blocks = new_block->vmm_segment->phy_blocks.size();
        new_block->vmm_segment->used_blocks = 0;

        large_blocks.blocks.insert(new_block);
            
        if(!get_fused_fragmented_blocks(p, 4)) {
          GMLAKE_INFO(" call get_fused_fragmented_blocks failed");
          return false;
        }
      } else {
        for(size_t i = 0; i < new_block->vmm_segment->phy_blocks.size(); i++) {
          new_block->vmm_segment->phy_blocks[i]->mapped_blocks.emplace_back(new_block, i);
          new_block->vmm_segment->phy_blocks[i]->free = false;
        }
            
        new_block->vmm_segment->free_blocks = 0;
        new_block->vmm_segment->used_blocks = new_block->vmm_segment->phy_blocks.size();
            
        p.block = new_block;
        p.err = cudaSuccess;
      }
    } else {
      p.block = new_block;
      p.err = cudaSuccess;
    }
    
    return true;
  }



  bool alloc_block(AllocParams& p, bool isRetry) {
    // Defensively checks for preexisting CUDA error state.
    C10_CUDA_CHECK(cudaGetLastError());

    size_t size = p.alloc_size;
    void* ptr;

    if (isRetry) {
      stats.num_alloc_retries += 1;
    }

    if (set_fraction &&
        total_allocated_memory + size > allowed_memory_maximum) {
      p.err = cudaErrorMemoryAllocation;
      return false;
    } else {
      p.err = cudaMallocMaybeCapturing(&ptr, size);
      if (p.err != cudaSuccess) {
        if (p.err == cudaErrorMemoryAllocation) {
          // If this is the first attempt (!isRetry), we can forgive and clear
          // CUDA's internal error state.
          //
          // If this is the second attempt (isRetry), malloc's TORCH_CHECK_WITH
          // will take over to throw a helpful exception. The user can choose
          // to catch the exception, free some stuff in their script, and
          // attempt the allocation again. In this case, we can also forgive and
          // clear CUDA's internal error state.
          cudaGetLastError();
        } else {
          // If the error's unrelated to memory allocation, we should throw
          // immediately.
          C10_CUDA_CHECK(p.err);
        }
        return false;
      }
    }

    if (p.pool->owner_PrivatePool) {
      // The block is for a CUDA graph's PrivatePool.
      p.pool->owner_PrivatePool->cudaMalloc_count++;
    }

    total_allocated_memory += size;
    p.block = new Block(p.device(), p.stream(), size, p.pool, (char*)ptr);
    for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
      update_stat(stats.segment[stat_type], 1);
      update_stat(stats.reserved_bytes[stat_type], size);
    });
    if (size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_segments, 1);

    // p.block came from new, not cudaMalloc. It should not be nullptr here.
    TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);
    return true;
  }

  /** Free one or more oversize blocks to the system allocator.  But only enough
   * **/
  /** to satisfy the target size **/
  bool release_available_cached_blocks(const AllocParams& p) {
    if (CachingAllocatorConfig::max_split_size() ==
        std::numeric_limits<size_t>::max())
      return false;
    BlockPool& pool = *p.pool;

    // because of std::unique_ptr, block cannot be trivially copied
    Block key(
        p.search_key.device,
        p.search_key.stream,
        p.search_key.size,
        p.search_key.pool,
        p.search_key.ptr);
    key.size = (key.size < CachingAllocatorConfig::max_split_size())
        ? CachingAllocatorConfig::max_split_size()
        : key.size;
    auto it = pool.blocks.lower_bound(&key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
      // No single block is large enough; free multiple oversize blocks,
      // starting with the largest
      if (it == pool.blocks.begin())
        return false;
      size_t totalReleased = 0;
      --it; // Back up one item.  Now on the largest block for the correct
            // stream
      while ((totalReleased < key.size) &&
             ((*it)->size >= CachingAllocatorConfig::max_split_size()) &&
             ((*it)->stream == p.stream())) {
        auto cur = it;
        totalReleased += (*it)->size;
        if (it != pool.blocks.begin()) {
          --it;
          release_block(*cur);
        } else {
          release_block(*cur);
          break;
        }
      }
      if (totalReleased < key.size)
        return false;
    } else {
      release_block(*it);
    }
    return true;
  }

  bool release_cached_blocks() {
    // First ensure that all blocks that can't currently be allocated due to
    // outstanding events are returned to the pool.
    synchronize_and_free_events();

    // Free all non-split cached blocks to system allocator
    release_blocks(large_blocks);
    release_blocks(small_blocks);

    for (auto it = graph_pools_freeable.begin();
         it != graph_pools_freeable.end();) {
      // See notifyCaptureDestroy for the strategy here.
      TORCH_INTERNAL_ASSERT(it->second->use_count == 0);
      release_blocks(it->second->small_blocks);
      release_blocks(it->second->large_blocks);
      if (it->second->cudaMalloc_count == 0) {
        auto erase_count = graph_pools.erase(it->first);
        TORCH_INTERNAL_ASSERT(erase_count == 1);
        it = graph_pools_freeable.erase(it);
      } else {
        ++it;
      }
    }

    return true;
  }

  void release_block(Block* block) {
    static const int vmmDefragment = ([]()->int{
        const char* env = getenv("vmmDefragment");
        if(env) return atoi(env);
        else return 1;
    })();



    if (vmmDefragment > 0 && block->vmm_segment) {
      for(size_t i=0; i < block->vmm_segment->phy_blocks.size(); i++) {
        auto& phy_block = block->vmm_segment->phy_blocks[i];
        if(!phy_block->free) {
          GMLAKE_INFO(" warning for non fused blocks has non free phy_block: %lu, something wrong happended, block %p, block->ptr %p, block->size %fMB, free_blocks %lu, used_blocks %lu, event_id: %lu",
                                               i, block, block->ptr, block->size/(1024.f*1024.f), block->vmm_segment->free_blocks, block->vmm_segment->used_blocks, block->self_last_event->event_id);
        }

        for(auto& block_segment : phy_block->mapped_blocks) {
          Block* other_block = block_segment.block;
              
          if(other_block == block) continue;
              
          if(other_block->vmm_segment->fused) {
            if(active_fused_blocks.count(other_block) && 
              active_fused_blocks_to_gc.count(other_block) == 0) {
              {
                auto tmp1 = std::move(other_block->vmm_segment->vir_blocks[block_segment.offset]);
                auto tmp2 = std::move(other_block->vmm_segment->phy_blocks[block_segment.offset]);
              }
                      
              //active_fused_blocks.erase(other_block);
              active_fused_blocks_to_gc.insert(other_block);
            } else if(free_fused_blocks.blocks.count(other_block) || 
                          fragmented_free_fused_blocks[other_block->stream].blocks.count(other_block)) {
              if(free_fused_blocks.blocks.count(other_block)) {
                free_fused_blocks.blocks.erase(other_block);
                free_fused_blocks_in_release_order[other_block->stream].erase(other_block);
              } else if(fragmented_free_fused_blocks[other_block->stream].blocks.count(other_block)) {
                fragmented_free_fused_blocks[other_block->stream].erase(other_block);
              }
       
              for(auto& phy_block : other_block->vmm_segment->phy_blocks) {
                int i = 0;
                for(int j = 0; j < phy_block->mapped_blocks.size(); j++) {
                  if(phy_block->mapped_blocks[j].block != other_block) {
                    if(i != j) {
                      phy_block->mapped_blocks[i] = phy_block->mapped_blocks[j];
                    }
                                  
                    i++;
                  }
                }
                phy_block->mapped_blocks.resize(i);
              }
                      
                      
              delete other_block;
            }
          } else {
            GMLAKE_INFO(" warning for non fused blocks has phy_block mapped to other non fused blocks");
            exit(-1);
          }
        }
                        
      }
      
      
    }
    
    

    
    if(block->vmm_segment){
      block->vmm_segment.reset();
    } else {
      C10_CUDA_CHECK(cudaFree((void*)block->ptr));
    }
    total_allocated_memory -= block->size;

    auto* pool = block->pool;
    if (pool->owner_PrivatePool) {
      // The cudaFreed block belonged to a CUDA graph's PrivatePool.
      TORCH_INTERNAL_ASSERT(pool->owner_PrivatePool->cudaMalloc_count > 0);
      pool->owner_PrivatePool->cudaMalloc_count--;
    }

    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*pool))] = true;
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.segment[stat_type], -1);
      update_stat(
          stats.reserved_bytes[stat_type],
          -static_cast<std::int64_t>(block->size));
    });
    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_segments, -1);
    if (block->history) {
      record_trace(
          TraceEntry::SEGMENT_FREE,
          int64_t(block->ptr),
          block->size,
          block->stream,
          block->history->h.context);
    }
    pool->blocks.erase(block);
    delete block;
  }

  void release_blocks(BlockPool& pool) {
    // Frees all non-split blocks
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
      Block* block = *it;
      ++it;
      if (!block->prev && !block->next) {
        release_block(block);
      }
    }
  }

  EventPool::Event create_event_internal(int idx) {
    // Leak the event pool to avoid shutdown issues.
    static auto* event_pool = new EventPool();
    return event_pool->get(idx);
  }

  void synchronize_and_free_events() {
    // Synchronize on outstanding events and then free associated blocks.

    // This function syncs, so capture should not be underway. Might as well
    // make sure capture-deferred end of life events get processed too.
    TORCH_INTERNAL_ASSERT(captures_underway == 0);
    insert_events_deferred_until_no_capture();

    for (auto& st : cuda_events) {
      for (auto& e : st.second) {
        EventPool::Event event = std::move(e.first);
        Block* block = e.second;

        C10_CUDA_CHECK(cudaEventSynchronize(*event));

        block->event_count--;
        if (block->event_count == 0) {
          // free_block(block);
          update_block(block);
        }
      }
    }

    cuda_events.clear();
  }

  void insert_events(Block* block) {
    int prev_device;
    C10_CUDA_CHECK(cudaGetDevice(&prev_device));

    stream_set streams(std::move(block->stream_uses));
    AT_ASSERT(block->stream_uses.empty());
    for (auto& stream : streams) {
      C10_CUDA_CHECK(cudaSetDevice(stream.device_index()));

      EventPool::Event event =
          create_event_internal(static_cast<int>(stream.device_index()));
      C10_CUDA_CHECK(cudaEventRecord(*event, stream.stream()));

      block->event_count++;
      cuda_events[stream].emplace_back(std::move(event), block);
    }

    C10_CUDA_CHECK(cudaSetDevice(prev_device));
  }

  void insert_free_event_into_alloc_stream(Block* block)
  {
    int prev_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&prev_device));
    if(prev_device != block->device)
    {
      C10_CUDA_CHECK(cudaSetDevice(block->device));
    }
  

    
    if( block->self_last_event && 
        block->self_last_event.unique() && 
        block->self_last_event->stream == block->stream && 
        !block->self_last_event->ref_as_sync) {
      block->self_last_event->record(block->stream);
    } else {
      block->self_last_event = std::make_shared<BlockEvent>(block->stream, true);
    }

    if(prev_device != block->device) {
      C10_CUDA_CHECK(cudaSetDevice(prev_device));
    }
  }

  void insert_events_deferred_until_no_capture() {
    if (C10_UNLIKELY(needs_events_deferred_until_no_capture.size() > 0)) {
      for (auto* block : needs_events_deferred_until_no_capture) {
        TORCH_INTERNAL_ASSERT(!block->stream_uses.empty());
        insert_events(block);
      }
      needs_events_deferred_until_no_capture.clear();
    }
  }

  void process_events() {
    insert_events_deferred_until_no_capture();

    // Process outstanding cudaEvents. Events that are completed are
    // removed from the queue, and the 'event_count' for the
    // corresponding allocation is decremented. We maintain a separate
    // list of events per stream to avoid head-of-line delays if one
    // or more streams has long-running operations.

    // Iterate over different streams.
    for (auto it = cuda_events.begin(); it != cuda_events.end();) {
      // Iterate over this stream's (event, block) pairs.
      while (!it->second.empty()) {
        auto& e = it->second.front();
        EventPool::Event event = std::move(e.first);
        Block* block = e.second;

        cudaError_t err = C10_CUDA_ERROR_HANDLED(cudaEventQuery(*event));
        if (err == cudaErrorNotReady) {
          // ignore and clear the error if not ready
          cudaGetLastError();
          // Return the ownership of the Event (unique ptr)
          e.first = std::move(event);
          break;
        } else if (err != cudaSuccess) {
          C10_CUDA_CHECK(err);
        }

        block->event_count--;
        if (block->event_count == 0) {
          // free_block(block);
          update_block(block);
        }
        it->second.pop_front();
      }

      if (it->second.empty()) {
        it = cuda_events.erase(it);
      } else {
        it++;
      }
    }
  }

  // Iterates over sizes of all memory blocks for given device in given pool
  void cache_info_aux(const BlockPool& pool, size_t* largest) {
    for (const auto& block : pool.blocks) {
      const auto blocksize = block->size;
      if (blocksize > *largest) {
        *largest = blocksize;
      }
    }
  }

  void record_trace(
      TraceEntry::Action action,
      int64_t addr,
      size_t size,
      cudaStream_t stream,
      std::shared_ptr<Context> context) {
    auto te = TraceEntry(
        action,
        addr,
        size,
        stream,
        alloc_trace_record_context_ ? std::move(context) : nullptr);
    if (alloc_trace->size() < alloc_trace_max_entries_) {
      alloc_trace->emplace_back(te);
    } else {
      (*alloc_trace)[alloc_trace_next++] = te;
      if (alloc_trace_next == alloc_trace_max_entries_) {
        alloc_trace_next = 0;
      }
    }
  }
};

// Returns whether to force all allocations to bypass the caching allocator and
// go straight to cudaMalloc.  This setting is useful when debugging GPU memory
// errors, since the caching allocator foils cuda-memcheck.
bool forceUncachedAllocator() {
  static bool force_uncached =
      getenv("PYTORCH_NO_CUDA_MEMORY_CACHING") != nullptr;
  return force_uncached;
}

static void uncached_delete(void* ptr) {
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_memory_deallocation(reinterpret_cast<uintptr_t>(ptr));
  }
  C10_CUDA_CHECK(cudaFree(ptr));
}

void local_raw_delete(void* ptr);

class NativeCachingAllocator : public CUDAAllocator {
 private:
  std::mutex mutex;

  // allocated blocks by device pointer
  ska::flat_hash_map<void*, Block*> allocated_blocks;

  void add_allocated_block(Block* block) {
    std::lock_guard<std::mutex> lock(mutex);
    allocated_blocks[block->ptr] = block;
  }

 public:
  std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocator;

  Block* get_allocated_block(void* ptr, bool remove = false) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      return nullptr;
    }
    Block* block = it->second;
    if (remove) {
      allocated_blocks.erase(it);
    }
    return block;
  }

  void init(int device_count) override {
    const auto size = static_cast<int64_t>(device_allocator.size());
    static const int vmmDefragment = ([]()->int{
        const char* env = getenv("vmmDefragment");
        if(env) return atoi(env);
        else return 1;
    })();
    if (vmmDefragment) {
        GMLAKE_INFO(" GMLAKE initialized");
    }
    if (size < device_count) {
      device_allocator.resize(device_count);
      for (const auto i : c10::irange(size, device_count)) {
        device_allocator[i] = std::make_unique<DeviceCachingAllocator>();
      }
    }
  }

  bool initialized() override {
    return device_allocator.size() > 0;
  }

  /** allocates a block which is safe to use from the provided stream */
  void malloc(void** devPtr, int device, size_t size, cudaStream_t stream) {
    TORCH_INTERNAL_ASSERT(
        0 <= device && static_cast<size_t>(device) < device_allocator.size(),
        "Allocator not initialized for device ",
        device,
        ": did you call init?");
    Block* block = device_allocator[device]->malloc(device, size, stream);
    add_allocated_block(block);
    *devPtr = (void*)block->ptr;
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_memory_allocation(
          reinterpret_cast<uintptr_t>(*devPtr));
    }
  }

  void free(void* ptr) {
    if (!ptr) {
      return;
    }
    Block* block = get_allocated_block(ptr, true /* remove */);
    if (!block) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_memory_deallocation(
          reinterpret_cast<uintptr_t>(block->ptr));
    }
    device_allocator[block->device]->free(block);
  }

  void setMemoryFraction(double fraction, int device) override {
    TORCH_INTERNAL_ASSERT(
        0 <= device && static_cast<size_t>(device) < device_allocator.size(),
        "Allocator not initialized for device ",
        device,
        ": did you call init?");
    TORCH_INTERNAL_ASSERT(
        0 <= fraction && fraction <= 1,
        "invalid fraction:",
        fraction,
        ". Please set within (0, 1).");
    int activated_device;
    C10_CUDA_CHECK(cudaGetDevice(&activated_device));
    if (activated_device != device) {
      C10_CUDA_CHECK(cudaSetDevice(device));
    }
    device_allocator[device]->setMemoryFraction(fraction);
  }

  void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      bool alloc_trace_record_context) override {
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    device_allocator[device]->recordHistory(
        enabled,
        std::move(context_recorder),
        alloc_trace_max_entries,
        alloc_trace_record_context);
  }

  void attachOutOfMemoryObserver(OutOfMemoryObserver observer) override {
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    device_allocator[device]->attachOutOfMemoryObserver(std::move(observer));
  }

  void emptyCache() override {
    for (auto& da : device_allocator)
      da->emptyCache();
  }

  void* getBaseAllocation(void* ptr, size_t* outSize) override {
    Block* block = get_allocated_block(ptr);
    if (!block) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    return device_allocator[block->device]->getBaseAllocation(block, outSize);
  }

  void recordStream(const DataPtr& ptr, cuda::CUDAStream stream) override {
    // Empty tensor's storage().data() might be a null ptr. As there is no
    // blocks associated with those tensors, it is fine to do nothing here.
    if (!ptr.get()) {
      return;
    }

    // If a tensor is not allocated by this instance, simply skip
    // This usually happens when CUDA tensors are shared across processes,
    // we have implemented reference counting based sharing mechanism to
    // guarantee tensors won't be accidentally freed by one process while
    // they are still being used in another
    if (ptr.get_deleter() != &local_raw_delete)
      return;

    Block* block = get_allocated_block(ptr.get());
    // block must not be null reaching here
    TORCH_INTERNAL_ASSERT(block != nullptr, "No allocated block can be found");
    device_allocator[block->device]->recordStream(block, stream);
  }

  SnapshotInfo snapshot() override {
    SnapshotInfo result;
    for (auto& da : device_allocator) {
      result.device_traces.emplace_back(da->trace());
      auto snap = da->snapshot();
      result.segments.insert(result.segments.end(), snap.begin(), snap.end());
    }
    return result;
  }
  DataPtr allocate(size_t size) const override {
    constexpr size_t one_exa_bytes = 1152921504606846976ULL;
    TORCH_CHECK_WITH(
        OutOfMemoryError,
        size < one_exa_bytes,
        "CUDA out of memory. Tried to allocate more than 1EB memory.");
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    void* r = nullptr;
    if (forceUncachedAllocator()) {
      // Deliberately don't use cudaMallocMaybeCapturing here, to force an error
      // if someone tries to use forceUncachedAllocator while capturing.
      C10_CUDA_CHECK(cudaMalloc(&r, size));
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_memory_allocation(reinterpret_cast<uintptr_t>(r));
      }
      return {r, r, &uncached_delete, Device(DeviceType::CUDA, device)};
    }
    if (size != 0) {
      // Allocator declars allocate const!?
      const_cast<NativeCachingAllocator*>(this)->malloc(
          &r, device, size, cuda::getCurrentCUDAStream(device));
    }
    return {r, r, &local_raw_delete, Device(DeviceType::CUDA, device)};
  }
  DeleterFnPtr raw_deleter() const override {
    if (forceUncachedAllocator()) {
      return &uncached_delete;
    } else {
      return &local_raw_delete;
    }
  }
  void cacheInfo(int dev_id, size_t* largestBlock) override {
    device_allocator[dev_id]->cacheInfo(largestBlock);
  }
  void assertValidDevice(int device) {
    const auto device_num = device_allocator.size();
    TORCH_CHECK(
        0 <= device && device < static_cast<int64_t>(device_num),
        "Invalid device argument ",
        device,
        ": did you call init?");
  }

  DeviceStats getDeviceStats(int device) override {
    assertValidDevice(device);
    return device_allocator[device]->getStats();
  }

  void resetAccumulatedStats(int device) override {
    assertValidDevice(device);
    device_allocator[device]->resetAccumulatedStats();
  }

  void resetPeakStats(int device) override {
    assertValidDevice(device);
    device_allocator[device]->resetPeakStats();
  }
  // CUDAGraph interactions
  void notifyCaptureBegin(
      int device,
      CaptureId_t graph_id,
      MempoolId_t mempool_id) override {
    assertValidDevice(device);
    device_allocator[device]->notifyCaptureBegin(
        graph_id, std::move(mempool_id));
  }

  void notifyCaptureAboutToEnd(int device, CaptureId_t graph_id) override {
    assertValidDevice(device);
    device_allocator[device]->notifyCaptureAboutToEnd(graph_id);
  }

  void notifyCaptureEnded(int device, CaptureId_t graph_id) override {} // no-op

  void notifyCaptureDestroy(int device, MempoolId_t mempool_id) override {
    assertValidDevice(device);
    device_allocator[device]->notifyCaptureDestroy(std::move(mempool_id));
  }

  void* raw_alloc(size_t nbytes) override {
    if (nbytes == 0) {
      return nullptr;
    }
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    void* r = nullptr;
    malloc(&r, device, nbytes, cuda::getCurrentCUDAStream(device));
    return r;
  }

  void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
    if (nbytes == 0) {
      return nullptr;
    }
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    void* r = nullptr;
    malloc(&r, device, nbytes, stream);
    return r;
  }
  bool needsPoolSpecificPeerAccess() override {
    return false;
  }
  void raw_delete(void* ptr) override {
    this->free(ptr);
  }

  // In CUDA IPC, sender sends a tensor to receiver, getIpcDevPtr
  // is called by the receiving process to map the CUDA memory from the sending
  // process into its own address space.
  //
  // CUDA IPC only allows sharing a big memory block associated with a
  // cudaIpcMemHandle_t and it can be opened only **once** per context per
  // process. There can be multiple types of storage in the same IPC mem block,
  // so we must cache the device ptr to construct typed storage as it comes.
  //
  // ipcMemHandle_to_devptr maps a cudaIpcMemHandle_t to a device pointer in the
  // process that can be used to access the memory block in the sender process.
  // It only saves a weak_ptr of the device pointer in the map, the shared_ptr
  // will be used to reconstruct all storages in this CudaMalloc allocation. And
  // it will deleted in cudaIpcCloseMemHandle when its reference count is 0.
  //
  std::mutex IpcMutex;
  ska::flat_hash_map<std::string, std::weak_ptr<void>> ipcMemHandle_to_devptr;
  std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
    std::lock_guard<std::mutex> lock(IpcMutex);

    auto iter = ipcMemHandle_to_devptr.find(handle);
    if (iter != ipcMemHandle_to_devptr.end()) {
      auto devptr = iter->second.lock();
      if (devptr)
        return devptr;
    }
    // This ipcMemHandle hasn't been opened, or already expired, open it to
    // enable IPC access to that mem block.
    void* dev = nullptr;
    auto ipc_handle =
        reinterpret_cast<const cudaIpcMemHandle_t*>(handle.c_str());
    C10_CUDA_CHECK(cudaIpcOpenMemHandle(
        &dev, *ipc_handle, cudaIpcMemLazyEnablePeerAccess));
    // devPtr has to be deleted in same device when created.
    int curr_device;
    C10_CUDA_CHECK(cudaGetDevice(&curr_device));
    auto sp =
        std::shared_ptr<void>(dev, [handle, curr_device, this](void* ptr) {
          cuda::CUDAGuard device_guard(curr_device);
          std::lock_guard<std::mutex> deleter_lock(IpcMutex);
          C10_CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
          ipcMemHandle_to_devptr.erase(handle);
        });
    std::weak_ptr<void> wp = sp;
    // To eliminate an additional search, we can use insert().
    // It doesn't overwrite when key already exists(ptr expired).
    // But in the deleter for sp we erased the entry,
    // this should be safe to do now.
    ipcMemHandle_to_devptr.insert(iter, {handle, wp});

    return sp;
  }
  std::string name() override {
    return "native";
  }
};

NativeCachingAllocator allocator;

void local_raw_delete(void* ptr) {
  allocator.free(ptr);
}

void setAllocatorSettings(const std::string& env) {
  CachingAllocatorConfig::instance().parseArgs(env.c_str());
}

} // namespace Native

// General caching allocator utilities
void setAllocatorSettings(const std::string& env) {
  CachingAllocatorConfig::instance().parseArgs(env.c_str());
}

// Size pretty-printer
inline std::string format_size(uint64_t size) {
  std::ostringstream os;
  os.precision(2);
  os << std::fixed;
  if (size <= 1024) {
    os << size << " bytes";
  } else if (size <= 1048576) {
    os << (size / 1024.0);
    os << " KiB";
  } else if (size <= 1073741824ULL) {
    os << size / 1048576.0;
    os << " MiB";
  } else {
    os << size / 1073741824.0;
    os << " GiB";
  }
  return os.str();
}

namespace CudaMallocAsync {
// If this is put in its own header file, it gets incorrectly renamed in HIPify.
CUDAAllocator* allocator();

} // namespace CudaMallocAsync

struct BackendStaticInitializer {
  // Parses env for backend at load time, duplicating some logic from
  // CachingAllocatorConfig. CachingAllocatorConfig double-checks it later (at
  // runtime). Defers verbose exceptions and error checks, including Cuda
  // version checks, to CachingAllocatorConfig's runtime doublecheck. If this
  // works, maybe we should move all of CachingAllocatorConfig here?
  CUDAAllocator* parseEnvForBackend() {
    const char* val = getenv("PYTORCH_CUDA_ALLOC_CONF");
    if (val != nullptr) {
      const std::string config(val);

      std::regex exp("[\\s,]+");
      std::sregex_token_iterator it(config.begin(), config.end(), exp, -1);
      std::sregex_token_iterator end;
      std::vector<std::string> options(it, end);

      for (auto option : options) {
        std::regex exp2("[:]+");
        std::sregex_token_iterator it2(option.begin(), option.end(), exp2, -1);
        std::sregex_token_iterator end2;
        std::vector<std::string> kv(it2, end2);
        if (kv.size() >= 2) {
          if (kv[0] == "backend") {
            if (kv[1] == "cudaMallocAsync")
              return CudaMallocAsync::allocator();
            if (kv[1] == "native")
              return &Native::allocator;
          }
        }
      }
    }
    return &Native::allocator;
  }

  BackendStaticInitializer() {
    auto r = parseEnvForBackend();
    allocator.store(r);
  }
};

std::atomic<CUDAAllocator*> allocator{};
BackendStaticInitializer backend_static_initializer;

} // namespace CUDACachingAllocator
} // namespace cuda
} // namespace c10
