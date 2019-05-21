"""Replaybuffer for DQN.

- Auther: Chaehyeuk Lee
- Contact: chaehyeuk.lee@medipixel.io
- Reference: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf (DQN)
             https://github.com/medipixel/practice_collaboration (Medipixel Github)
             https://github.com/higgsfield/RL-Adventure (RL Adventure)
"""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, batch_size, device):
        self.buffer: list = list()
        self.buffer_size = capacity
        self.batch_size = batch_size  
        self.index = 0
        self.device = device

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)

        if len(self.buffer) == self.buffer_size:
            # If buffer is full, the oldest data is replaced by new data sequantially.
            self.buffer[self.index] = data
            self.index = (self.index + 1) % self.buffer_size
        else:
            self.buffer.append(data)

    def sample(self):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        indices = np.random.choice(
            len(self.buffer), size=self.batch_size, replace=False
        )

        for i in indices:
            s, a, r, n_s, d = self.buffer[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            rewards.append(np.array(r, copy=False))
            next_states.append(np.array(n_s, copy=False))
            dones.append(np.array(float(d), copy=False))

        # Change sampled data to torch tensor variable
        states_ = torch.FloatTensor(np.array(states)).to(self.device)
        actions_ = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards_ = torch.FloatTensor(np.array(rewards).reshape(-1, 1)).to(self.device)
        next_states_ = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_ = torch.FloatTensor(np.array(dones).reshape(-1, 1)).to(self.device)

        # Cuda available
        # Returns a copy of this object in CUDA memory. 
        # If 'True' and the source is in pinned memory, 
        # the copy will be asynchronous with respect to the host.
        if torch.cuda.is_available():
            states_ = states_.cuda(non_blocking=True)
            actions_ = actions_.cuda(non_blocking=True)
            rewards_ = rewards_.cuda(non_blocking=True)
            next_states_ = next_states_.cuda(non_blocking=True)
            dones_ = dones_.cuda(non_blocking=True)

        return states_, actions_, rewards_, next_states_, dones_

    def __len__(self):
        return len(self.buffer)
