# rl_model.py
import torch
import torch.nn as nn

class MyPolicy(nn.Module):
    def __init__(self, state_dim=12, action_dim=6):
        super(MyPolicy, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # normalizes actions to [-1,1], adjust if your env rescales
        )

    def forward(self, x):
        return self.actor(x)
