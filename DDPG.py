'''
Author: Owen Mo
Implementation of a DDPG agent
'''
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils import average_weights
from torch_utils.builder import build_fc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_config = {
    "latent_dim": 1024,
    "discount": 0.99,
    "lr": 3e-4,
    "explore_eps": 0.1,
    "tau": 0.99,
}

class DDPG():
    '''
    A DDPG agent
    State Space: Continuous
    Action Space: Continuous
    '''
    def __init__(self, state_dim, action_dim, min_action=-1, max_action=1, config=None):
        # state_dim should be a tuple with one element
        self.state_dim = state_dim
        # action_dim should be a tuple with one element
        self.action_dim = action_dim
        if config is None:
            config = default_config.copy()
        self.discount = config["discount"]
        self.eps = config["explore_eps"]
        self.tau = config["tau"]
        self.min_action = min_action
        self.max_action = max_action
        self._build_networks(config["latent_dim"], config["lr"])


    def _build_networks(self, latent_dim, lr):
        self.actor = nn.Sequential(
            build_fc(self.state_dim[0], latent_dim),
            build_fc(latent_dim, latent_dim),
            build_fc(latent_dim, self.action_dim[0], activation='none'),
        )
        self.critic = nn.Sequential(
            build_fc(self.state_dim[0]+self.action_dim[0], latent_dim),
            build_fc(latent_dim, latent_dim),
            build_fc(latent_dim, 1, activation='none'),
        )
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
    
    def _scale_action(self, v):
        scaled_v = (torch.sigmoid(v) * (self.max_action-self.min_action)) + self.min_action
        return scaled_v

    def select_action(self, state):
        action = self.actor(torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device))
        action = self._scale_action(action)
        action = action.squeeze(0).detach().cpu().numpy()
        return action + np.random.normal(0, self.eps, self.action_dim)

    def train(self, replay_buffer, batch_size=100):
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        not_done = 1 - done
        
        with torch.no_grad():
            next_actions = self._scale_action(self.target_actor(next_state))
            next_values = self.target_critic(torch.cat([next_state, next_actions], dim=-1))
        
        # Critic always has output of size 1 (the value)
        target_values = reward + not_done * self.discount * next_values.squeeze(-1) 

        # Critic update
        self.critic_optim.zero_grad()
        current_values = self.critic(torch.cat([state, action], axis=-1)).squeeze(-1)
        mse = nn.MSELoss()
        critic_loss = mse(current_values, target_values)
        
        critic_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor_optim.zero_grad()
        actor_loss = -self.critic(torch.cat([state, self._scale_action(self.actor(state))], dim=-1)).mean()

        actor_loss.backward()
        self.actor_optim.step()
    
        # Update target network if needed
        self.target_actor.load_state_dict(average_weights(self.target_actor.state_dict(), self.actor.state_dict(), self.tau))
        self.target_critic.load_state_dict(average_weights(self.target_critic.state_dict(), self.critic.state_dict(), self.tau))
