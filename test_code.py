"""Test code for DQN model and ReplayBuffer on "Acrobot-v1" environment.

- Auther: Chaehyeuk Lee
- Contact: chaehyeuk.lee@medipixel.io
"""

# import numpy as np
import torch
import torch.optim as optim
import gym

from model.dqn import DQN
from buffer.replaybuffer import ReplayBuffer

BUFFER_SIZE = 5
BATCH_SIZE = 2
GAMMA = 0.99
EPSILON = 0.3

env = gym.make("Acrobot-v1")
state = env.reset()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device)
model = DQN(env)

if torch.cuda.is_available():
    model = model.cuda()

optimizer = optim.Adam(model.parameters())

# Replay buffer test code
for _ in range(BUFFER_SIZE):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    memory.add(state, action, reward, next_state, done)
    state = next_state

replay = memory.sample()
print(replay)
print("All replay buffer Length : ", memory.__len__())  # This should be same with BUFFER_SIZE.
print("Sampled states: ", replay[0])  # This should print the same number of states with batch.
print("One of sampled state: ", replay[0][0])  # This should print only one state set.

# Select action test code
for _ in range(5):
    action = model.select_action(replay[0][0], EPSILON)
    print("Action : ", action)  # action should be same except random case.

# Forward Calculation test code
forward_result = model.forward(replay[0][0])
print("Foward Result 1 : ", forward_result)
forward_result = model.forward(replay[0][1])
print("Foward Result 2 : ", forward_result)
# The dimension should be same with the number of action.

# Check if tensor is working properly.
forward_result = model.forward(replay[0])
print("Foward Result 1 + 2: ", forward_result)