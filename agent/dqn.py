"""Agent class for DQN.

- Auther: Chaehyeuk Lee
- Contact: chaehyeuk.lee@medipixel.io
- Reference: https://github.com/medipixel/rl_algorithms (Medipixel Github)
             https://github.com/higgsfield/RL-Adventure (RL Adventure)
"""

import numpy as np
import torch
import torch.nn.functional as F
import gym

from typing import Tuple


class DQNAgent:
    """Agent methods class for DQN.

    Attributes:
            env (gym.Env): OpenAI gym environment
            model (Tuple): Neural Network model of DQN
            model_target (Tuple): Target Neural Network model of DQN
            gamma (float): decay parameter
            device (torch.device): device used in this program (cpu or cuda)
    """

    def __init__(
        self,
        env: gym.Env,
        model: Tuple,
        model_target: Tuple,
        memory: Tuple,
        gamma: float,
        device: torch.device
    ):
        """Initialize a DQNAgent object.

        Args:
            env (gym.Env): OpenAI gym environment
            model (tuple): Neural Network model of DQN
            model_target (Tuple): Target Neural Network model of DQN
            memory (Tuple): Replay buffer memory for DQN
            gamma (float): decay parameter
            device (torch.device): device used in this program (cpu or cuda)
        """

        self.env = env
        self.model = model
        self.model_target = model_target
        self.memory = memory
        self.gamma = gamma
        self.device = device

    def select_action(self, state: np.ndarray, epsilon: float, is_learn: bool = True) -> int:
        """Select action based on Q-value and Epsilon-greedy method.

        Args:
            state (numpy.ndarray): state from gym environment
            epsilon (float): epsilon-greedy exploration parameter
            is_learn (bool): select action with exploration or not
        """
        if is_learn and np.random.random() <= epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.FloatTensor(state).to(self.device)
            q_value = self.model.layers(state)
            action = torch.argmax(q_value, dim=0).item()

        return action

    def get_td_loss(self) -> torch.Tensor:
        """Compute and return dqn loss."""
        states, actions, rewards, next_states, dones = self.memory.sample()

        next_q_values = self.model_target.layers(next_states)
        next_max_q_values = torch.max(next_q_values, dim=1)[0]
        target_q = rewards + self.gamma * next_max_q_values * (1 - dones)
        target_q = target_q.detach()

        q_values = self.model.layers(states)
        estimated_q = torch.gather(q_values, dim=1, index=actions.unsqueeze(dim=1))
        # estimated_q = q_set[torch.arange(q_set.size(0)), actions_]

        # loss = F.smooth_l1_loss(estimated_q, target_q)
        loss = (estimated_q - target_q).pow(2).mean()

        return loss

    def decay_epsilon(self, epsilon, min_epsilon, max_epsilon, epsilon_decay):
        return max(epsilon - (max_epsilon - min_epsilon) * epsilon_decay, min_epsilon)
