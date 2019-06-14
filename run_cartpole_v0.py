"""Train algorithm for DQN on "Cartpole-v0" environment.

- Auther: Chaehyeuk Lee
- Contact: chaehyeuk.lee@medipixel.io
"""

import random

import gym
import numpy as np
import torch
import torch.optim as optim

from config.dqn import HyperParams
import wandb
from agent.dqn import DQNAgent
from buffer.replaybuffer import ReplayBuffer
from model.dqn import DQN


# Env initialize
env = gym.make("CartPole-v0")

# Seeding
env.seed(2)
torch.manual_seed(2)
np.random.seed(2)
random.seed(2)

# Model, Buffer, torch initialize
cuda_on = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_on else "cpu")
memory = ReplayBuffer(HyperParams["BUFFER_SIZE"], HyperParams["BATCH_SIZE"], device)
model = DQN(env)
model_target = DQN(env)
optimizer = optim.Adam(model.parameters(), lr=HyperParams["LEARNING_RATE"])
agent = DQNAgent(
    env, model, model_target, optimizer, memory, HyperParams["GAMMA"], device
)

if cuda_on:
    model = model.cuda()
    model_target = model_target.cuda()

# Initialize wandb
if HyperParams["IS_LOG"] is True:
    wandb.init(config=HyperParams)

# Copy model parameter to target parameter
model_target.load_state_dict(model.state_dict())
total_step = 0
epsilon = HyperParams["MAX_EPSILON"]

for episode in range(HyperParams["MAX_EPISODE"]):
    reward_log = np.array([])
    state = env.reset()
    score = 0
    while True:
        # env.render()

        action = agent.select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        memory.add(state, action, reward, next_state, done)
        state = next_state

        # Update network
        loss = 0.0
        if len(memory) >= HyperParams["BATCH_SIZE"]:
            loss = agent.update_model()

        total_step += 1
        score += reward

        # Epsilon decay
        epsilon = agent.decay_epsilon(
            epsilon,
            HyperParams["MIN_EPSILON"],
            HyperParams["MAX_EPSILON"],
            HyperParams["EPSILON_DECAY"],
        )

        # Update target network
        if (
            total_step >= HyperParams["TARGET_UPDATE_AFTER"]
            and total_step % HyperParams["TARGET_UPDATE_STEP"] == 0
        ):
            model_target.load_state_dict(model.state_dict())

        if done is True:
            agent.write_log(episode, score, loss, epsilon, HyperParams["IS_LOG"])
            break
