"""Test code for DQN model and ReplayBuffer on "CartPole-v0" environment.

- Auther: Chaehyeuk Lee
- Contact: chaehyeuk.lee@medipixel.io
"""

import gym
import torch
import torch.optim as optim

from buffer.replaybuffer import ReplayBuffer
from model.dqn import DQN

BUFFER_SIZE = 5
BATCH_SIZE = 2
GAMMA = 0.99
EPSILON = 0.3

env = gym.make("CartPole-v0")
state = env.reset()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device)
model = DQN(env, device)

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

# This should be same with BUFFER_SIZE.
print("All replay buffer Length : ", memory.__len__())
# This should print the same number of states with batch.
print("Sampled states: ", replay[0])
# This should print only one state set.
print("One of sampled state: ", replay[0][0])

# Select action test code
for _ in range(5):
    action = model.select_action(replay[0][0], EPSILON)
    print("Action : ", action)  # action should be same except random case.

# Forward Calculation test code
# The dimension should be same with the number of action.
forward_result = model.forward(replay[0][0])
print("Foward Result 1 : ", forward_result)
forward_result = model.forward(replay[0][1])
print("Foward Result 2 : ", forward_result)

# Check if np2tensor is working properly.
forward_result = model.forward(replay[0])
print("Foward Result 1 + 2: ", forward_result)

# Check if tensor forward calculation is working properly.
q_tensor = model.forward_tensor(replay[0])
print("Forward Tensor Result : ", q_tensor)

# Check action tensor
action_tensor = replay[1]
print("Action Tensor Result :", action_tensor)

# Check q value of desired action
q_by_action_tensor = q_tensor[torch.arange(q_tensor.size(0)), action_tensor]
print("Selected Q-Value : ", q_by_action_tensor)

# get max q_value tensor
max_q_tensor = model.get_max_q(replay[0])
print("Max Q Tensor : ", max_q_tensor)
