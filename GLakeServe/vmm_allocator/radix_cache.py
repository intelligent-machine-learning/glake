import heapq
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple

import sys
sys.path.append('/usr/lib64')
import vmmAllocator

import torch


class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.value = []
        self.ref_counter = 0
        self.last_access_time = time.time()
        self.prefix_len = 0

    def __lt__(self, other):
        return self.last_access_time < other.last_access_time


def match(key, seq):
    i = 0
    for k, w in zip(key, seq):
        if k != w:
            break
        i += 1
    return i

class radixDict:
    def __init__(self, token_size):
        self.token_size = token_size
        self.key = []
        self.value = []
    
    def match_prefix(self, key):
        if len(key) == 0 or len(self.key) == 0:
            return None, 0
        else:
            index = 0
            max_prefix_len = 0
            key = tuple(key)
            for i in range(len(self.key)):
                k = self.key[i]
                prefix_len = match(k, key)
                if max_prefix_len < prefix_len:
                    index = i
                    max_prefix_len = prefix_len
            value = self.value[index]
            self.value.pop(index)
            self.key.pop(index)
            return value, max_prefix_len
    
    def insert(self, key, value=None):
        key_len = len(key)
        key = tuple(key)
        self.key.append(key)
        self.value.append(value)
        return key_len
    
    def get_free_slot(self,):
        index = len(self.key)
        if index == 0:
            return None
        value = self.value[index - 1]
        self.key.pop(index - 1)
        self.value.pop(index - 1)

class RadixCache:
    def __init__(self, token_size, disable=False):
        self.root_node = TreeNode()
        self.root_node.value = []
        self.root_node.ref_counter = 1
        self.evictable_size_ = 0

        self.token_size = token_size

        self.disable = disable

    ##### Public API #####
    def match_prefix(self, key):
        if self.disable:
            return [], self.root_node

        value = []
        key = tuple(key)
        last_node = [self.root_node]
        self._match_prefix_helper(self.root_node, key, value, last_node, 0)
        if len(value) == 0:
            return None, 0
        else:
            return last_node[0].value[0], last_node[0].prefix_len

    def insert(self, key, value=None):
        if self.disable:
            return len(key)
        key = tuple(key)
        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value, 0)

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper(self.root_node)

    def evict(self, num_tokens, evict_callback):
        if self.disable:
            raise RuntimeError()

        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.ref_counter > 0:
                continue

            num_evicted += evict_callback(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

    def inc_ref_counter(self, node):
        delta = 0
        while node != self.root_node:
            if node.ref_counter == 0:
                self.evictable_size_ -= len(node.value)
                delta -= len(node.value)
            node.ref_counter += 1
            node = node.parent
        return delta

    def dec_ref_counter(self, node):
        delta = 0
        while node != self.root_node:
            if node.ref_counter == 1:
                self.evictable_size_ += len(node.value)
                delta += len(node.value)
            node.ref_counter -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    ##### Internal Helper Functions #####
    def _match_prefix_helper(self, node, key, value, last_node, p_len):
        node.last_access_time = time.time()

        for c_key, child in node.children.items():
            prefix_len = match(c_key, key)
            if prefix_len != 0:
                tmp_p_len = prefix_len + p_len
                if prefix_len < len(c_key):
                    new_node = self._split_node(c_key, child, prefix_len, tmp_p_len)
                    value.append(new_node.value)
                    last_node[0] = new_node
                else:
                    last_node[0] = child
                    value.append(child.value)
                    self._match_prefix_helper(child, key[prefix_len:], value, last_node, tmp_p_len)
                break

    def _split_node(self, key, child, split_len, p_len):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {key[split_len:]: child}
        new_node.parent = child.parent
        new_node.ref_counter = child.ref_counter
        new_node.prefix_len = p_len
        new_node.value.append(vmmAllocator.split_and_copy(child.value[0], self.token_size * p_len))
        child.parent = new_node
        new_node.parent.children[key[:split_len]] = new_node
        del new_node.parent.children[key]
        return new_node

    def _insert_helper(self, node, key, value, p_len):
        node.last_access_time = time.time()
        
        for c_key, child in node.children.items():
            prefix_len = match(c_key, key)

            if prefix_len == len(c_key):
                if prefix_len == len(key):
                    # child.value.append(value)
                    return prefix_len
                else:
                    key = key[prefix_len:]
                    tmp_p_len = prefix_len + p_len
                    return prefix_len + self._insert_helper(child, key, value, tmp_p_len)

            if prefix_len:
                new_node = self._split_node(c_key, child, prefix_len, prefix_len + p_len)
                return prefix_len + self._insert_helper(
                    new_node, key[prefix_len:], value, prefix_len + p_len
                )

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.value.append(value)
            new_node.prefix_len = p_len + len(key)
            node.children[key] = new_node
            #self.evictable_size_ += len(value)
        return 0

    def _print_helper(self, node, indent):
        for key, child in node.children.items():
            print(" ", len(key), key[:20])
            for kv_segment in child.value:
                print("value is")
                kv_segment.read_string()
            print("prefix_len is ", child.prefix_len)
            self._print_helper(child, indent=indent + 2)

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(k)

    def _total_size_helper(self, node):
        x = len(node.value)
        for child in node.children.values():
            x += self._total_size_helper(child)
        return x

    def _collect_leaves(self):
        ret_list = []

        def dfs_(cur_node):
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)

            for x in cur_node.children.values():
                dfs_(x)

        dfs_(self.root_node)
        return ret_list


if __name__ == "__main__":
    tree = RadixCache(1, disable=False)

    vmmAllocator.init_physical_handle(10)

    kv_segment1 = vmmAllocator.KVcacheSegment(1, 2097152, 0)
    kv_segment1.write_string("Hello")
    #print("kv_segment1 str is")
    #kv_segment1.read_string()

    tree.insert("Hello", kv_segment1)

    kv_segment2 = vmmAllocator.KVcacheSegment(1, 2097152, 0)
    kv_segment2.write_string("Hello")
    #print("kv_segment2 str is")
    #kv_segment2.read_string()

    #tree.insert("Hello", kv_segment2)

    kv_segment3 = vmmAllocator.KVcacheSegment(1, 2097152, 0)
    kv_segment3.write_string("Hello_L.A.!")
    #print("kv_segment3 str is")
    #kv_segment3.read_string()

    tree.insert("Hello_L.A.!", kv_segment3)
    
    print("print tree info")
    tree.pretty_print()

    kv_segment, prefix_len = tree.match_prefix("Hell")
    print("match prefix_len is ", prefix_len)
    kv_segment.read_string()

    #tree.insert("Hello")
    #tree.insert("Hello")
    #tree.insert("Hello_L.A.!")
    #tree.insert("Hello_world! Happy")
    #tree.insert("I love you!")
    #tree.pretty_print()

    # print(tree.match_prefix("I love you! aha"))

    # def evict_callback(x):
    #    print("evict", x)
    #    return len(x)

    # tree.evict(5, evict_callback)
    # tree.evict(10, evict_callback)
    # tree.pretty_print()
