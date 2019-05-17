import numpy as np
import torch
import torch.optim as optim
import gym

from model import dqnmodel
from buffer import replaybuffer

BUFFER_SIZE = 1000
BATCH_SIZE = 128
GAMMA = 0.99
EPSILON = 0.1

env = gym.make('Acrobot-v1')
env.reset()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

memory = replaybuffer.ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device)

model = dqnmodel.DQN(env)

if torch.cuda.is_available():
    model = model.cuda()

optimizer = optim.Adam(model.parameters())