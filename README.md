# Train-Robots-konnex
the simulator clean up kitchen
fetch-ai-policy
│
├── policy.py
├── example_run.py
├── requirements.txt
└── README.md
📄 policy.py
import torch
import torch.nn as nn
import numpy as np


class FetchPolicy(nn.Module):
    """
    Baseline policy for Fetch robot control (13D action output)
    Compatible with Konnex Fetch Interface
    """

    def __init__(self, obs_dim, action_dim=13):
        super(FetchPolicy, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, obs):
        return self.model(obs)

    def act(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            action = self.forward(obs)

        return action.numpy()


if __name__ == "__main__":
    obs_dim = 30
    policy = FetchPolicy(obs_dim)

    dummy_obs = np.random.rand(obs_dim)
    action = policy.act(dummy_obs)

    print("Generated Action:", action)
📄 example_run.py
import numpy as np
from policy import FetchPolicy

def main():
    obs_dim = 30  # Adjust based on your environment
    policy = FetchPolicy(obs_dim)

    dummy_observation = np.random.rand(obs_dim)
    action = policy.act(dummy_observation)

    print("Observation:", dummy_observation)
    print("Action Output (13D):", action)


if __name__ == "__main__":
    main()
📄 requirements.txt
torch
numpy
📄 README.md
# Fetch AI Policy — Konnex Compatible

Baseline AI control policy for the Fetch mobile manipulator robot compatible with the Konnex Fetch Interface.

## Overview

This model generates a 13-dimensional action vector used to control:

- 7 arm joints
- 1 gripper
- 3 head/torso controls
- 2 base movements

Output values are normalized between [-1, 1].

## Features

- Lightweight neural network
- Compatible with Fetch robot simulation
- Easy to extend for Reinforcement Learning
- Ready for Konnex AI Miner submission

## Installation

```bash
pip install -r requirements.txt
Usage
python example_run.py
Model Input
Observation vector (size configurable)

Example: robot joint states, positions, velocities

Model Output
13D action vector

Range: [-1, 1]

Future Improvements
PPO/SAC training

Vision-based input

Task-conditioned policies






---

