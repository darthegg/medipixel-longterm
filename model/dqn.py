"""DQN model for simple environmnets without CNN.

- Auther: Chaehyeuk Lee
- Contact: chaehyeuk.lee@medipixel.io
- Reference: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf (DQN)
             https://github.com/medipixel/rl_algorithms (Medipixel Github)
             https://github.com/higgsfield/RL-Adventure (RL Adventure)
"""

import gym
import torch.nn as nn


class DQN(nn.Module):
    """Affine neural network model for DQN on Acrobot-v1 environment.

    Taken from RL Adventure.
    https://github.com/higgsfield/RL-Adventure

    Attributes:
        env (gym.Env): OpenAI gym environment
        layers (torch.nn.Sequential): Sequential Neural Network model
    """

    def __init__(self, env: gym.Env):
        """Initialize a ReplayBuffer object.

        Args:
            env (gym.Env): OpenAI gym environment
        """
        super(DQN, self).__init__()
        self.env = env

        self.layers = nn.Sequential(
            nn.Linear(self.env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.env.action_space.n),
        )
