import numpy as np
import random
import math

class Node:

    def __init__(self, value, data):
        self.value = value 
        self.data = data

class PrioritizedMemory:

    def __init__(self, maxlen, epsilon=0.01, alpha=0.5):
        self.cur_pos = 0
        self.maxlen = maxlen
        self.epsilon = epsilon
        self.alpha = alpha
        self.root_count = self._size_by_depth(math.ceil(math.log2(self.maxlen))-1) 
        self.tree = [Node(0, None) for i in range(maxlen + self.root_count)]

    def _size_by_depth(self, depth):
        return sum([2 ** i for i in range(depth+1)])

    def _propogate(self, index, value_change):
        parent_index = math.ceil(index / 2) - 1
        self.tree[parent_index].value += value_change
        
        if parent_index == 0:
            return

        self._propogate(parent_index, value_change)

    def insert(self, error, sample, compute_priority=True):
        if compute_priority:
            priority = (error + self.epsilon) ** self.alpha
        else:
            priority = error

        index = self.root_count + self.cur_pos    
        old_priority = self.tree[index].value
        self.tree[index].value = priority
        self.tree[index].data = sample

        self._propogate(index, priority - old_priority)

        self.cur_pos += 1
        if self.cur_pos == self.maxlen:
            self.cur_pos = 0

    def _is_leaf(self, index):
        if index >= self.root_count:
            return True
        
        return False

    def _left(self, index):
        return index * 2 + 1
    
    def _right(self, index):
        return index * 2 + 2

    def _retrieve(self, index, value):
        if self._is_leaf(index):
            return self.tree[index].data 

        if self.tree[self._left(index)].value >= value:
            return self._retrieve(self._left(index), value)
        
        else:
            return self._retrieve(self._right(index), value - self.tree[self._left(index)].value)
        
    def retrieve(self):
        rand_num = random.uniform(0, self.tree[0].value)
        return self._retrieve(0, rand_num)

    def __str__(self):
        return ''.join([str(i.value) + ', ' for i in self.tree])