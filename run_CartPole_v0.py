"""Train algorithm for DQN on "Cartpole-v0" environment.

- Auther: Chaehyeuk Lee
- Contact: chaehyeuk.lee@medipixel.io
"""

# import numpy as np
import torch
import torch.optim as optim
import gym
import numpy as np

from model.dqn import DQN
from buffer.replaybuffer import ReplayBuffer


BUFFER_SIZE = 10000
BATCH_SIZE = 128
GAMMA = 0.99
epsilon = 0.01
max_epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.0001
EPOCH = 20

MAX_EPISODE = 5
MAX_STEP = 500


# Env initialize
env = gym.make("CartPole-v0")
state = env.reset()

# Model, buffer, torch initialize
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device)
model = DQN(env, device)

if torch.cuda.is_available():
    model = model.cuda()

optimizer = optim.Adam(model.parameters())

reward_log = np.zeros((MAX_STEP, 1))
episode = 0


while True:
    for _ in range(MAX_EPISODE):
        for i in range(MAX_STEP):
            env.render()
            action = model.select_action(state, epsilon, 'train')
            next_state, reward, done, _ = env.step(action)
            memory.add(state, action, reward, next_state, done)
            state = next_state
            reward_log[i] = reward
            if done is True:
                break
        state = env.reset()

        epsilon = max(epsilon - (max_epsilon - min_epsilon) * epsilon_decay, min_epsilon)
        episode = episode + 1

        print("episode : ", episode,  "reward average : ", np.average(reward_log))

        reward_log = np.zeros((MAX_STEP, 1))

    model_target = model

    for _ in range(EPOCH):
        states_, actions_, rewards_, next_states_, dones_ = memory.sample()

        next_max_q = model_target.get_max_q(next_states_)
        measured_q = rewards_ + GAMMA * next_max_q * (1 - dones_)

        q_set = model.forward_tensor(states_)
        estimated_q = q_set[torch.arange(q_set.size(0)), actions_]

        loss = (measured_q - estimated_q).pow(2).mean()
        # print("Loss : ", loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # TODO
    # save and load model
    # train stability (not converge now)
    # make methods (train, get_loss)




