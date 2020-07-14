'''
Author: Owen Mo
Implementation of a DQN agent
'''
import random
import torch
import torch.nn as nn
import torch.optim as optim

from torch_utils.builder import build_fc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_config = {
    "latent_dim": 1024,
    "discount": 0.99,
    "lr": 3e-4,
    "explore_eps": 0.1
}

class DDPG():
    '''
    A DDPG agent
    State Space: Continuous
    Action Space: Continuous
    '''
    def __init__(self, state_dim, action_dim, config=None):
        # state_dim should be a tuple with one element
        self.state_dim = state_dim
        # action_dim should be a tuple with one element
        self.action_dim = action_dim


    def _build_networks(self, latent_dim):
        self.actor = nn.Sequential(
            build_fc(self.state_dim[0], latent_dim),
            build_fc(latent_dim, latent_dim),
            build_fc(latent_dim, self.action_dim[0])
        )
        self.critic = nn.Sequential(
            build_fc(self.state_dim[0]+self.action_dim[0], latent_dim),
            build_fc(latent_dim, latent_dim),
            build_fc(latent_dim, 1)
        )
        self.
