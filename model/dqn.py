"""DQN model for simple environmnets without CNN.

- Auther: Chaehyeuk Lee
- Contact: chaehyeuk.lee@medipixel.io
- Reference: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf (DQN)
             https://github.com/medipixel/rl_algorithms (Medipixel Github)
             https://github.com/higgsfield/RL-Adventure (RL Adventure)
"""

import numpy as np
import torch
import torch.nn as nn
import gym


class DQN(nn.Module):
    """Affine neural network model for DQN on Acrobot-v1 environment.

    Taken from RL Adventure.
    https://github.com/higgsfield/RL-Adventure

    Attributes:
        env (gym.Env): OpenAI gym environment
        layers (torch.nn.Sequential): Sequential Neural Network model
    """

    def __init__(self, env: gym.Env, device: torch.device):
        """Initialize a ReplayBuffer object.

        Args:
            env (gym.Env): OpenAI gym environment
        """
        super(DQN, self).__init__()
        self.env = env
        self.device = device

        self.layers = nn.Sequential(
            nn.Linear(self.env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.env.action_space.n)
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forwad calculate neural network model (one state).

        Args:
            x (numpy.ndarray): state from gym environment
        """
        x = torch.tensor(x, device=self.device).float()
        return self.layers(x)

    def forward_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Forwad calculate neural network model (states set(tensor)).

        Args:
            x (torch.Tensor): sampled states
        """
        return self.layers(x)

    def select_action(self, state: np.ndarray, epsilon: float, test: str) -> int:
        """Select action based on Q-value and Epsilon-greedy method.

        Args:
            state (numpy.ndarray): state from gym environment
            epsilon (float): epsilon-greedy exploration parameter
        """
        if test == 'train':
            if np.random.random() > epsilon:
                q_value = self.forward(state)
                _, action = q_value.max(0)
                action = action.item()
            else:
                action = self.env.action_space.sample()
                # print("!!!!!!!!!!!!!! Random Action !!!!!!!!!!!!!!!")
            return action

        else:
            q_value = self.forward(state)
            _, action = q_value.max(0)
            action = action.item()

            return action

    def get_max_q(self, states_: torch.Tensor) -> torch.Tensor:
        """Get Maimum Q-Value

        Args:
            state_ (torch.Tensor): state from gym environment
        """
        q_value = self.forward_tensor(states_)
        # print("Q values : ", q_value)
        max_q_value, _ = q_value.max(1)
        return max_q_value

    # TODO
    # def loss_calculate
    # def train
