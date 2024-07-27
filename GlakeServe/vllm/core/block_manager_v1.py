"""A block manager that manages token blocks."""
import math
from abc import ABC, abstractmethod
from itertools import count, takewhile
from os.path import commonprefix
from typing import Dict, List, Optional
from typing import Sequence as GenericSequence
from typing import Set, Tuple

from vllm.block import BlockTable, PhysicalTokenBlock
from vllm.core.evictor_v1 import EvictionPolicy, Evictor, make_evictor
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device

logger = init_logger(__name__)


class BlockAllocatorBase(ABC):
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    @abstractmethod
    def __init__(self,
                 device: Device,
                 block_size: int,
                 num_blocks: int,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        pass

    @abstractmethod
    def allocate(self,
                 block_hash: Optional[int] = None,
                 num_hashed_tokens: int = 0) -> PhysicalTokenBlock:
        pass

    @abstractmethod
    def free(self, block: PhysicalTokenBlock) -> None:
        pass

    @abstractmethod
    def get_num_free_blocks(self) -> int:
        pass

    @abstractmethod
    def get_num_total_blocks(self) -> int:
        pass

    @abstractmethod
    def contains_block(self, block_hash: int) -> bool:
        pass

    @abstractmethod
    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        pass


class CachedBlockAllocator(BlockAllocatorBase):
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(self,
                 device: Device,
                 block_size: int,
                 num_blocks: int,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU) -> None:
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks

        self.current_num_blocks = 0
        self.cached_blocks: Dict[int, PhysicalTokenBlock] = {}

        self.evictor: Evictor = make_evictor(eviction_policy)

        self.default_hash_ctr = count()

    def allocate_block(self, block_hash: int,
                       num_hashed_tokens: int) -> PhysicalTokenBlock:
        if self.current_num_blocks == self.num_blocks:
            block = self.evictor.evict()
            block.block_hash = block_hash
            block.num_hashed_tokens = num_hashed_tokens
            return block
        block = PhysicalTokenBlock(device=self.device,
                                   block_number=self.current_num_blocks,
                                   block_size=self.block_size,
                                   block_hash=block_hash,
                                   num_hashed_tokens=num_hashed_tokens)
        self.current_num_blocks += 1
        return block

    def allocate(self,
                 block_hash: Optional[int] = None,
                 num_hashed_tokens: int = 0) -> PhysicalTokenBlock:
        if block_hash is None:
            block_hash = next(self.default_hash_ctr)
        if block_hash in self.evictor:
            assert block_hash not in self.cached_blocks
            block = self.evictor.remove(block_hash)
            assert block.ref_count == 0
            self.cached_blocks[block_hash] = block
            block.ref_count += 1
            assert block.block_hash == block_hash
            return block
        if block_hash not in self.cached_blocks:
            self.cached_blocks[block_hash] = self.allocate_block(
                block_hash, num_hashed_tokens)
        block = self.cached_blocks[block_hash]
        assert block.block_hash == block_hash
        block.ref_count += 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            assert block.block_hash not in self.evictor
            self.evictor.add(block)

            # Remove the block from the cached_blocks
            del self.cached_blocks[block.block_hash]

    def get_num_free_blocks(self) -> int:
        return (self.num_blocks - self.current_num_blocks +
                self.evictor.num_blocks)

    def get_num_total_blocks(self) -> int:
        return self.num_blocks

    def contains_block(self, block_hash: int) -> bool:
        return block_hash in self.cached_blocks or block_hash in self.evictor

    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        # Update the hash of block and the cached_blocks dictionary.
        assert not self.contains_block(block_hash)
        old_hash = block.block_hash
        block.block_hash = block_hash
        del self.cached_blocks[old_hash]
        self.cached_blocks[block_hash] = block


class UncachedBlockAllocator(BlockAllocatorBase):
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        device: Device,
        num_gpu_handles: int,
    ) -> None:
        self.device = device
        self.num_gpu_handles = num_gpu_handles

        self.free_blocks = num_gpu_handles

    def allocate(self,
                 num: int) -> None:
        if self.free_blocks < num:
            raise ValueError("Out of memory! No free blocks are available.")
        self.free_blocks = self.free_blocks - num

    def free(self, num: int) -> None:
        self.free_blocks += num

    def get_num_free_blocks(self) -> int:
        return self.free_blocks

    def get_num_total_blocks(self) -> int:
        return self.num_gpu_handles

    def contains_block(self, block_hash: int) -> bool:
        raise NotImplementedError(
            "Invalid codepath for uncached block allocator.")

    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        raise NotImplementedError(
            "Invalid codepath for uncached block allocator.")


class BlockSpaceManagerV1(BlockSpaceManager):
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        max_seqs: int,
        block_size: int,
        num_gpu_handles: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
        cache_token: int = 0,
        cache_seq: int = 0, 
    ) -> None:
        self.max_seqs = max_seqs
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_handles
        self.num_total_cpu_blocks = num_cpu_blocks

        if enable_caching and sliding_window is not None:
            raise NotImplementedError(
                "Sliding window is not allowed with prefix caching enabled!")

        self.block_sliding_window = None
        if sliding_window is not None:
            # Round up to nearest block size to regularize sliding window
            # allocation sizes.
            self.block_sliding_window = math.ceil(sliding_window / block_size)

        self.watermark = watermark
        assert watermark >= 0.0

        self.enable_caching = enable_caching

        self.watermark_blocks = int(watermark * num_gpu_handles)

        if self.enable_caching:
            logger.info("Automatic prefix caching is enabled.")
            self.gpu_allocator: BlockAllocatorBase = CachedBlockAllocator(
                Device.GPU, block_size, num_gpu_handles)
            self.cpu_allocator: BlockAllocatorBase = CachedBlockAllocator(
                Device.CPU, block_size, num_cpu_blocks)
        else:
            self.gpu_allocator = UncachedBlockAllocator(
                Device.GPU, num_gpu_handles)
            self.cpu_allocator = UncachedBlockAllocator(
                Device.CPU, num_cpu_blocks)
        # Mapping: seq_id -> BlockTable.
        self.block_tables: Dict[int, int] = {}

        self.kv_cache_slot = []
        for i in range(self.max_seqs):
            self.kv_cache_slot.append(0)

        self.cache_token = cache_token
        self.cache_seq = cache_seq
        for i in range(self.max_seqs):
            self.block_tables[i] = cache_token * 2 if i < cache_seq else 0 
        self.gpu_allocator.allocate(cache_seq * cache_token * 2)
        self.free_block_tables_block = cache_seq * cache_token * 2 # chunks num cached inside block_tables
        
    def get_free_slot(self) -> int:
        for i in range(self.max_seqs):
            if self.kv_cache_slot[i] == 0:
                self.kv_cache_slot[i] = 1
                return i
        return -1

    def get_cached_slot(self) -> int:
        for i in range(self.max_seqs):
            if self.kv_cache_slot[i] == 0 and self.block_tables[i] == self.cache_token * 2:
                self.kv_cache_slot[i] = 1
                self.free_block_tables_block -= self.cache_token * 2
                return i
        
        return -1

    def garbage_collect(self) -> None:
        pass

    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_required_blocks = seq.get_logical_block_num()
        #if preempted_flag:
        #    num_required_blocks += 2
       
        # 1. have slot with enough chunks
            
        # 2. have slot but need allocation
            # a. if enough

            # b. not enough

        if self.block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks,
                                      self.block_sliding_window)
        minus = self.cache_token * 2 if self.free_block_tables_block > 0 else 0
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks() + minus

        # Use watermark to avoid frequent cache eviction.
        if (self.num_total_gpu_blocks - num_required_blocks <
                self.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def allocate(self, seq_group: SequenceGroup, slot: int) -> None:
        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        
        slot = self.get_cached_slot() # try get a cached slot
        if slot == -1:
            
            slot = self.get_free_slot() # get a empty slot
            if slot == -1:
                return False
            minus = 0
        minus = self.block_tables[slot]
        seq.set_slot(slot)
        seq_reserved_block_num = seq.get_logical_block_num() - minus
        if seq_reserved_block_num > 0:
            self.gpu_allocator.allocate(seq_reserved_block_num)
            self.block_tables[slot] += seq_reserved_block_num
        #print("allocate seq: ", seq.get_slot(), " ", seq.get_logical_block_num(), " ", seq_reserved_block_num)
        return True

       # # Allocate new physical token blocks that will store the prompt tokens.
       # num_prompt_blocks = len(seq.logical_token_blocks)

       # block_table: BlockTable = []
       # for logical_idx in range(num_prompt_blocks):
       #     if (self.block_sliding_window is not None
       #             and logical_idx >= self.block_sliding_window):
       #         block = block_table[logical_idx % self.block_sliding_window]
       #         # Set the reference counts of the token blocks.
       #         block.ref_count = seq_group.num_seqs()
       #     elif self.enable_caching:
       #         block = self.gpu_allocator.allocate(
       #             seq.hash_of_block(logical_idx),
       #             seq.num_hashed_tokens_of_block(logical_idx))
       #     else:
       #         block = self.gpu_allocator.allocate()
       #         # Set the reference counts of the token blocks.
       #         block.ref_count = seq_group.num_seqs()
       #     block_table.append(block)

       # # Assign the block table for each sequence.
       # for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
       #     self.block_tables[seq.seq_id] = block_table.copy()

    def can_append_slots(self,
                         seq_group: SequenceGroup,
                         num_lookahead_slots: int = 0,
                         beta: int = 0,) -> bool:
        assert (num_lookahead_slots == 0
                ), "lookahead allocation not supported in BlockSpaceManagerV1"
        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return 2 * num_seqs <= num_free_gpu_blocks + beta
    
    def cached_token_gc(self) -> List[int]:
        assert (gc_num <= self.free_block_tables_block // (self.cache_token * 2))
        gc_list = []
        gc_num = -num_free_gpu_blocks + 2 * num_seqs
        for i in range(self.max_seqs):
            if len(gc_list) < gc_num and self.kv_cache_slot[i] == 0 and self.block_tables[i] == self.cache_token * 2:
                self.free_block_tables_block -= self.cache_token * 2
                self.gpu_allocator.free(self.cache_token * 2)
                self.block_tables[i] -= self.cache_token * 2 
                gc_list.append(i)
    def _promote_last_block(
        self,
        seq: Sequence,
        last_block: PhysicalTokenBlock,
    ) -> PhysicalTokenBlock:
        assert self.enable_caching

        # Compute a new hash for the block so that it can be shared by other
        # Sequences
        new_hash = seq.hash_of_block(len(seq.logical_token_blocks) - 1)

        # if new_hash is already in the cached table, then free last_block
        # and return the cached version
        if self.gpu_allocator.contains_block(new_hash):
            self.gpu_allocator.free(last_block)
            return self.gpu_allocator.allocate(new_hash)
        else:
            self.gpu_allocator.update_hash(new_hash, last_block)
            return last_block

    def _is_last_block_full(
        self,
        seq: Sequence,
    ) -> bool:
        token_ids_len = seq.data.get_len()
        return token_ids_len > 0 and token_ids_len % seq.block_size == 0

    def _maybe_promote_last_block(
        self,
        seq: Sequence,
        last_block: PhysicalTokenBlock,
    ) -> PhysicalTokenBlock:
        if self._is_last_block_full(seq):
            return self._promote_last_block(seq, last_block)
        else:
            return last_block

    def _allocate_last_physical_block(
        self,
        seq: Sequence,
    ) -> PhysicalTokenBlock:
        # Called before a new block is appended.
        # This is in charge of allocating a new physical block (to be appended).

        # None if the last block is not full. Otherwise, we set it to the
        # content hash.
        if not self.enable_caching:
            return self.gpu_allocator.allocate()
        block_hash: Optional[int] = None
        if (self._is_last_block_full(seq)):
            block_hash = seq.hash_of_block(len(seq.logical_token_blocks) - 1)
        num_hashed_tokens = seq.num_hashed_tokens_of_block(
            len(seq.logical_token_blocks) - 1)

        # num_hashed_tokens is used to compute future hashes
        # (e.g. in the hashing function, it is used to ask the sequence for
        # prefix tokens)
        new_block = self.gpu_allocator.allocate(block_hash, num_hashed_tokens)

        # If the block has is None, then the block is not full.
        # If the block is not full, then we expect it to have a refcount of 1.
        if block_hash is None:
            assert new_block.ref_count == 1
        return new_block
    def sum_all_blocks(self):
        return sum(self.block_tables.values()) + self.gpu_allocator.get_num_free_blocks()
    
    def free_blocks(self):
        return self.gpu_allocator.get_num_free_blocks()

    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int = 0,
        extend_seq_list: List[int]=None, 
    ) -> List[Tuple[int, int]]:
        """Allocate a physical slot for a new token."""
        #return None
        seq_logical_block_num = seq.get_logical_block_num()
        seq_physical_block_num = self.block_tables[seq.get_slot()]
        # If we need to allocate a new physical block
      #  print(seq_physical_block_num, seq_logical_block_num)
        if seq_physical_block_num < seq_logical_block_num:
            # Currently this code only supports adding one physical block
            assert seq.get_slot() != -1
            extend_seq_list.append(seq.get_slot())
            #print(seq.get_slot(), "-", self.block_tables[seq.get_slot()], ":", self.gpu_allocator.get_num_free_blocks())
            #print("eager extend: ", extend_seq_list)
            self.gpu_allocator.allocate(2)
            self.block_tables[seq.get_slot()] = seq_physical_block_num + 2
    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        # NOTE: fork does not allocate a new physical block.
        # Thus, it is always safe from OOM.
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.copy()
        # When using a sliding window, blocks will be eventually reused.
        # In this case the block tables will contain repeated blocks.
        # When forking, we must make sure that each block's `ref_count`
        # is only incremented by one, so we deduplicate them by wrapping
        # them in a set.
        for block in set(src_block_table):
            block.ref_count += 1

    def _get_physical_blocks(
            self, seq_group: SequenceGroup) -> List[PhysicalTokenBlock]:
        # NOTE: Here, we assume that the physical blocks are only shared by
        # the sequences in the same group.
        blocks: Set[PhysicalTokenBlock] = set()
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            blocks.update(self.block_tables[seq.seq_id])
        return list(blocks)

    def can_swap_in(self,
                    seq_group: SequenceGroup,
                    num_lookahead_slots: int = 0) -> AllocStatus:
        assert (num_lookahead_slots == 0
                ), "BlockSpaceManagerV1 does not support lookahead allocation"
        blocks = self._get_physical_blocks(seq_group)
        num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
        num_free_blocks = self.gpu_allocator.get_num_free_blocks()
        # NOTE: Conservatively, we assume that every sequence will allocate
        # at least one free block right after the swap-in.
        # NOTE: This should match the logic in can_append_slot().
        num_required_blocks = len(blocks) + num_swapped_seqs
        if self.gpu_allocator.get_num_total_blocks() < num_required_blocks:
            return AllocStatus.NEVER
        elif num_free_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def swap_in(self,
                seq_group: SequenceGroup,
                num_lookahead_slots: int = 0) -> List[Tuple[int, int]]:
        assert (num_lookahead_slots == 0
                ), "BlockSpaceManagerV1 does not support lookahead allocation"

        # CPU block -> GPU block.
        # dict is efficient in lookup `if cpu_block in mapping`
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for cpu_block in block_table:
                if cpu_block in mapping:
                    gpu_block = mapping[cpu_block]
                    gpu_block.ref_count += 1
                else:
                    gpu_block = self.gpu_allocator.allocate(
                        cpu_block.block_hash, cpu_block.num_hashed_tokens)
                    mapping[cpu_block] = gpu_block
                new_block_table.append(gpu_block)
                # Free the CPU block swapped in to GPU.
                self.cpu_allocator.free(cpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            cpu_block.block_number: gpu_block.block_number
            for cpu_block, gpu_block in mapping.items()
        }
        # convert to list of tuples once here
        return list(block_number_mapping.items())

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        return len(blocks) <= self.cpu_allocator.get_num_free_blocks()

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        # GPU block -> CPU block.
        # dict is efficient in lookup `if gpu_block in mapping`
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for gpu_block in block_table:
                if gpu_block in mapping:
                    cpu_block = mapping[gpu_block]
                    cpu_block.ref_count += 1
                else:
                    cpu_block = self.cpu_allocator.allocate(
                        gpu_block.block_hash, gpu_block.num_hashed_tokens)
                    mapping[gpu_block] = cpu_block
                new_block_table.append(cpu_block)
                # Free the GPU block swapped out to CPU.
                self.gpu_allocator.free(gpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            gpu_block.block_number: cpu_block.block_number
            for gpu_block, cpu_block in mapping.items()
        }
        # convert to list of tuples once here
        return list(block_number_mapping.items())

    def _free_block_table(self, block_table: BlockTable) -> None:
        # when using a sliding window, each seq will only use up
        # to `self.block_sliding_window` blocks. When freeing
        # the block table, we must make sure to not free blocks more
        # than once. If no sliding window is used, there is no block
        # reuse in the block table, so we must free all blocks.
        blocks_to_free = (block_table[-self.block_sliding_window:]
                          if self.block_sliding_window is not None else
                          block_table)
        for block in set(blocks_to_free):
            if block.device == Device.GPU:
                self.gpu_allocator.free(block)
            else:
                self.cpu_allocator.free(block)
    
    #

    def free(self, seq: Sequence) -> int:
        if seq.get_slot() not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return -1
        slot = seq.get_slot()
        self.kv_cache_slot[slot] = 0
        seq_reserved_block_num = self.block_tables[slot]
        if (not seq.preempted):
            minus = 2*self.cache_token if seq_reserved_block_num >= 2*self.cache_token else 0
            blocks_to_free = seq_reserved_block_num - minus
            self.gpu_allocator.free(blocks_to_free)
            if minus == 2*self.cache_token:
                self.free_block_tables_block += 2*self.cache_token
            self.block_tables[slot] -= blocks_to_free
            #print("free seq: ", slot, " ", seq_reserved_block_num, " ",blocks_to_free)
        else: 
            self.gpu_allocator.free(seq_reserved_block_num)
            self.block_tables[slot] -= seq_reserved_block_num
            #print("free seq: ", slot, " ", seq_reserved_block_num)
        # del self.block_tables[seq.seq_id]
        return slot

    def reset(self) -> None:
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_table = self.block_tables[seq.seq_id]
        return [block.block_number for block in block_table]

    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        return self.cpu_allocator.get_num_free_blocks()

    def access_all_blocks_in_seq(
        self,
        seq: Sequence,
        access_time: float,
    ) -> None:
        if self.enable_caching:
            # Update the last accessed time of all the blocks accessed
            # in this step.
            block_table = self.block_tables[seq.get_slot()]
            for block in block_table:
                block.last_accessed = access_time

    def compute_full_blocks_in_seq(self, seq: Sequence):
        if seq.get_slot() not in self.block_tables:
            return
        max_full_block = seq.get_len() // self.block_size - 1
        block_table = self.block_tables[seq.get_slot()]
        if max_full_block == -1:
            return
        for i in reversed(range(max_full_block)):
            if block_table[i].computed:
                break
            block_table[i].computed = True

    def get_all_computed_blocks(self, seq: Sequence) -> List[int]:
        if seq.get_slot() not in self.block_tables:
            return []
        block_table = self.block_tables[seq.get_slot()]
        # NOTE We exclude the last block to avoid the case where the entire
        # prompt is cached. This would cause erroneous behavior in model
        # runner.
        return [
            b.block_number
            for b in takewhile(lambda b: b.computed, block_table[:-1])
        ]

    def get_common_computed_block_ids(
            self, seqs: List[Sequence]) -> GenericSequence[int]:
        """Return the block ids that are common for a given sequence group.

        Used in prefill (can skip prefill of some blocks).
        """
        # Can return non-empty result only with prefix caching enabled.
        if not self.enable_caching:
            return []

        ids_list = [self.get_all_computed_blocks(seq) for seq in seqs]
        return commonprefix([ids for ids in ids_list if ids != []])

    def mark_blocks_as_computed(self, seq_group: SequenceGroup):
        if self.enable_caching:
            for seq in seq_group.seqs_dict.values():
                self.compute_full_blocks_in_seq(seq)
