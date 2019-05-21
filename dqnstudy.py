"""Train algorithm for DQN on "Acrobot-v1" environment.

- Auther: Chaehyeuk Lee
- Contact: chaehyeuk.lee@medipixel.io
"""

# import numpy as np
import torch
import torch.optim as optim
import gym

from model.dqnmodel import DQN
from buffer.replaybuffer import ReplayBuffer

BUFFER_SIZE = 1000
BATCH_SIZE = 128
GAMMA = 0.99
EPSILON = 0.1

env = gym.make("Acrobot-v1")
env.reset()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device)

model = DQN(env)

if torch.cuda.is_available():
    model = model.cuda()

optimizer = optim.Adam(model.parameters())