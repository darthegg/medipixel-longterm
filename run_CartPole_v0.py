"""Train algorithm for DQN on "Cartpole-v0" environment.

- Auther: Chaehyeuk Lee
- Contact: chaehyeuk.lee@medipixel.io
"""

# import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import gym
import numpy as np

from model.dqn import DQN
from buffer.replaybuffer import ReplayBuffer


def get_td_loss(
    memory,
    model,
    model_target,
    gamma: float,
) -> torch.Tensor:
    """Compute and return dqn loss."""
    states, actions, rewards, next_states, dones = memory.sample()

    next_q_values = model_target(next_states)
    next_max_q_values = torch.max(next_q_values, dim=1)[0]
    target_q = rewards + gamma * next_max_q_values * (1 - dones)
    target_q = target_q.detach()

    q_values = model(states)
    estimated_q = torch.gather(q_values, dim=1, index=actions.unsqueeze(dim=1))
    # estimated_q = q_set[torch.arange(q_set.size(0)), actions_]

    # loss = F.smooth_l1_loss(estimated_q, target_q)
    loss = (estimated_q - target_q).pow(2).mean()

    return loss


BUFFER_SIZE = 1_000
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 2.5e-4
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.0005

MAX_EPISODE = 1000
TARGET_UPDATE_AFTER = 50
TARGET_UPDATE_STEP = 10
IS_LEARN = True

torch.manual_seed(0)

# Env initialize
env = gym.make("CartPole-v0")

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

total_step = 0

for episode in range(MAX_EPISODE):
    reward_log = np.array([])
    state = env.reset()
    score = 0
    while True:
        # env.render()

        action = model.select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        memory.add(state, action, reward, next_state, done)
        state = next_state
        
        # update network
        loss = 0.
        if len(memory) >= BATCH_SIZE:
            loss = get_td_loss(memory, model, model_target, GAMMA)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()

        total_step += 1
        score += reward

        if total_step >= TARGET_UPDATE_AFTER and total_step % TARGET_UPDATE_STEP == 0:
            model_target.load_state_dict(model.state_dict())

        if done is True:
            print(
                f"episode : {episode}\tscore : {score}",
                "\tLoss : %.4f" % (loss),
                "\tEpsilon : %.4f" % (epsilon))
            break

        epsilon = max(epsilon - (max_epsilon - min_epsilon) * epsilon_decay, min_epsilon)

