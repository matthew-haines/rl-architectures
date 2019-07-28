import numpy as np
import random

class Node:

    def __init__(self, value, data):
        self.value = value 
        self.data = data
        self.left = None
        self.right = None
        self.parent = None

    def is_leaf(self):
        if self.data == None:
            return False

        return True

    def is_root(self):
        if self.parent == None:
            return True
        
        return False


class PrioritizedMemory:

    def __init__(self, maxlen, epsilon=0.01, alpha=0.5):
        self.maxlen = maxlen
        self.epsilon = epsilon
        self.alpha = alpha
        self.head = Node(0, None)

    def _insert(self, node, sample):
        if 

    def insert(self, error, sample):
        priority = (error + self.epsilon) ** self.alpha
        node = Node(priority, sample)
        self._insert(self.head, node)