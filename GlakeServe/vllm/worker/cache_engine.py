"""CacheEngine class for managing the KV cache."""
from typing import List, Dict, Tuple

import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig
from vllm.logger import init_logger
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_pin_memory_available
from vmm_allocator.vmm_allocator import KV_cache
import sys
sys.path.append(r"/root/glake-vllm-open/vmm_allocator")
import vmmAllocator

logger = init_logger(__name__)

KVCache = Tuple[KV_cache, KV_cache]

class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        num_physical_handle: int,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_handles = cache_config.num_gpu_handles
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        
        dtype_size = _get_dtype_size(self.dtype)
        phy_gran = 2 * 1024 * 1024
        token_size = self.num_kv_heads * self.head_size * dtype_size
        max_context_size = self.scheduler_config.max_num_batched_tokens * token_size
        max_context_size = phy_gran * ((max_context_size + phy_gran - 1) // phy_gran)
        self.vmm_kvcache_segment = []
        if scheduler_config.cache_seqs > 0:
            #cache_seq_num = scheduler_config.cache_seqs
            #cache_token_size = scheduler_config.cache_token_size * token_size
            cache_token_size = phy_gran * scheduler_config.cache_tokens
            #print("max-num-seqs", self.scheduler_config.max_num_seqs)
            #print("cache-seqs", self.scheduler_config.cache_seqs)
            #print("max-context-size", max_context_size,self.scheduler_config.max_num_batched_tokens)
            for index in range(self.scheduler_config.max_num_seqs):
                self.vmm_kvcache_segment.append(vmmAllocator.KVcacheSegment(self.num_layers, max_context_size, cache_token_size, index < self.scheduler_config.cache_seqs)) 
        else:
            # cache all seqs
            for _ in range(self.scheduler_config.max_num_seqs):
                self.vmm_kvcache_segment.append(vmmAllocator.KVcacheSegment(self.num_layers, max_context_size, 0, False))

        # Get attention backend.
        self.attn_backend = get_attn_backend(
            model_config.get_num_attention_heads(parallel_config),
            self.head_size,
            self.num_kv_heads,
            model_config.get_sliding_window(),
            model_config.dtype,
            cache_config.cache_dtype,
            self.block_size,
        )

        # Initialize the cache.
        self.kv_batch_stride = None
        self.gpu_cache = self._allocate_kv_cache(self.num_gpu_handles, "cuda")
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

        k_cache = []
        v_cache = []
        for kv_cache in self.gpu_cache:
            k, v = kv_cache
            k_cache.append(k.kv_cache)
            v_cache.append(v.kv_cache)
        vmmAllocator.copy_kv_cache(k_cache, v_cache, self.vmm_kvcache_segment)

        # get kvcache device ptr
        self.k_cache_ptr = []
        self.v_cache_ptr = []
        for index in range(self.num_layers):
            k_ptr = []
            v_ptr = []
            for kv_cache_segment in self.vmm_kvcache_segment:
                k_ptr.append(kv_cache_segment.getKeyDevicePtr(index))
                v_ptr.append(kv_cache_segment.getValueDevicePtr(index))
            k_tensor_ptr = torch.tensor(k_ptr, dtype=torch.long, device='cuda')
            v_tensor_ptr = torch.tensor(v_ptr, dtype=torch.long, device='cuda')
            self.k_cache_ptr.append(k_tensor_ptr)
            self.v_cache_ptr.append(v_tensor_ptr)

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[KVCache]:
        """Allocates KV cache on the specified device."""
        kv_cache: List[KVCache] = []
        max_context_len = self.scheduler_config.max_num_batched_tokens
        max_seqs = self.scheduler_config.max_num_seqs
        dtype_size = torch.tensor([], dtype=self.dtype).element_size()
        phy_gran = 2 * 1024 * 1024
        max_context_size = max_context_len * self.num_kv_heads * self.head_size * dtype_size
        max_context_size = phy_gran * ((max_context_size + phy_gran - 1) // phy_gran)
        self.kv_batch_stride = max_context_size
        for layer_index in range(self.num_layers):
            key_blocks = KV_cache(max_seqs, layer_index)
            value_blocks = KV_cache(max_seqs, layer_index)
            kv_cache.append((key_blocks, value_blocks))
        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = _get_dtype_size(dtype)
        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
