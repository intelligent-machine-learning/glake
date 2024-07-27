import sys
sys.path.append(r"/usr/lib64")
import vmmAllocator

class KV_cache:
    """ref of the vmmAllocator KVCache
    """

    def __init__(
        self,
        max_seqs,
        layer_index: int,
    ) -> None:
        
        self.max_seqs = max_seqs
        self.layer_index = layer_index
        
        self.kv_cache = vmmAllocator.KVCache(self.max_seqs)
        self.vmm_ptr = self.kv_cache.getKVCachePtr()
