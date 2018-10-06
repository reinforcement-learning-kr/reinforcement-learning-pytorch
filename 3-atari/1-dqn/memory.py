import random
from collections import namedtuple

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('history', 'next_history', 'action', 'reward', 'mask'))


class Memory(object):
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.position = 0

    def push(self, history, next_history, action, reward, mask):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(history, next_history, action, reward, mask))
        self.memory[self.position] = Transition(history, next_history, action, reward, mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)