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

"""
# ep 500
BUFFER_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 0.0003
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.002

# PRE_EPISODE = 1
MAX_STEP = 100
MAX_EPISODE = 10000
TARGET_UPDATE_EP = 20
"""

BUFFER_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 0.0003
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.002

# PRE_EPISODE = 1
MAX_STEP = 100
MAX_EPISODE = 10000
TARGET_UPDATE_EP = 20



# Env initialize
env = gym.make("CartPole-v0")
state = env.reset()

# Model, buffer, torch initialize
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device)
model = DQN(env, device)
model_target = DQN(env, device)

if torch.cuda.is_available():
    model = model.cuda()
    model_target = model_target.cuda()

model_target.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def get_td_loss(memory, model, model_target, GAMMA):
    states_, actions_, rewards_, next_states_, dones_ = memory.sample()

    next_max_q = model_target.get_max_q(next_states_)
    measured_q = rewards_ + GAMMA * next_max_q * (1 - dones_)

    q_set = model.forward_tensor(states_)
    estimated_q = q_set[torch.arange(q_set.size(0)), actions_]

    loss = (measured_q - estimated_q).pow(2).mean()

    return loss

"""
for _ in range(PRE_EPISODE):
    while True:
        # env.render()
        
        action = model.select_action(state, epsilon, 'test')
        next_state, reward, done, _ = env.step(action)
        memory.add(state, action, reward, next_state, done)
        state = next_state
        if done is True:
            break
    state = env.reset()
"""

for episode in range(MAX_EPISODE):
    reward_log = np.array([])
    t = 0
    while True:
        #env.render()
        
        action = model.select_action(state, epsilon, 'train')
        next_state, reward, done, _ = env.step(action)
        memory.add(state, action, reward, next_state, done)
        state = next_state
        reward_log = np.append(reward_log, reward)
        
        loss = get_td_loss(memory, model, model_target, GAMMA)
        # print("Loss : ", loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t = t + 1

        if done is True:
            break

    if episode % TARGET_UPDATE_EP == 0:
        model_target.load_state_dict(model.state_dict())

    epsilon = max(epsilon - (max_epsilon - min_epsilon) * epsilon_decay, min_epsilon)
    print("episode : ", episode,  "reward sum : ", np.average(reward_log)*t/100, "Epsilon : ", epsilon)

    state = env.reset()


    # TODO
    # save and load model
    # train stability (not converge now)
    # make methods (train, get_loss)




