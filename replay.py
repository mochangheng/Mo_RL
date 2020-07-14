from collections import deque
from collections.abc import Sequence
import random
import torch

class ReplayBuffer(Sequence):
    def __init__(self, maxlen=2000):
        self.Q = deque(maxlen=maxlen)

    def __getitem__(self, index):
        return self.Q[index]

    def __len__(self):
        return len(self.Q)

    def add(self, item):
        self.Q.append(item)

    def sample(self, batch_size):
        batch = random.sample(self, batch_size)
        state = []
        action = []
        next_state = []
        reward = []
        done = []
        for s, a, next_s, r, d in batch:
            state.append(s)
            action.append(a)
            next_state.append(next_s)
            reward.append(r)
            done.append(d)
        
        # Returns all float tensors
        return (
            torch.tensor(state, dtype=torch.float),
            torch.tensor(action, dtype=torch.float),
            torch.tensor(next_state, dtype=torch.float),
            torch.tensor(reward, dtype=torch.float),
            torch.tensor(done, dtype=torch.float)
        )