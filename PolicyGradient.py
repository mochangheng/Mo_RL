'''
Author: Owen Mo
Implementation of a Policy Gradient agent
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

class PolicyGradient():
    '''
    A DQN agent
    State Space: Continuous
    Action Space: Discrete
    '''
    def __init__(self, state_dim, action_dim, config=None):
        # state_dim should be a tuple with one element
        self.state_dim = state_dim
        # action_dim should be an int
        self.action_dim = action_dim
        if config is None:
            config = default_config
        self.discount = config["discount"]
        self.eps = config["explore_eps"]
        self._build_networks(config["latent_dim"], config["lr"])

    def _build_networks(self, latent_dim, lr):
        self.Q_net = nn.Sequential(
            build_fc(self.state_dim[0], latent_dim),
            build_fc(latent_dim, latent_dim),
            build_fc(latent_dim, self.action_dim)
        )
        self.Q_optimizer = optim.Adam(self.Q_net.parameters(), lr=lr)

    def select_action(self, state):
        if random.random() < self.eps:
            # Choose random action from discrete action space
            action = random.randrange(self.action_dim)
        else:
            # Make greedy action
            values = self.Q_net(torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device))
            action = torch.argmax(values.squeeze()).item()

        return action

    def train(self, replay_buffer, batch_size=100):
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        # Discrete actions
        action = action.long()
        not_done = 1 - done
        
        with torch.no_grad():
            next_Q = self.Q_net(next_state)
        
        target_values = reward + not_done * self.discount * torch.max(next_Q, 1).values
        
        current_Q = self.Q_net(state)
        target_Q = current_Q.clone()
        target_Q[range(target_Q.size(0)), action] = target_values

        # Compute loss
        mse = nn.MSELoss()
        loss = mse(current_Q, target_Q)

        # Optimize Q function
        self.Q_optimizer.zero_grad()
        loss.backward()
        self.Q_optimizer.step()


        