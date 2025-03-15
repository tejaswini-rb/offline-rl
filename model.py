import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np
import random

# Q-Network Definition
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        # Concatenate state and action, then produce a single Q-value
        return self.fc(torch.cat([state, action], dim=1))

# Policy Network Definition
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, state):
        # Returns actions in [-1, 1]
        return torch.tanh(self.fc(state))

class CQLAgent:
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 lr=3e-4, 
                 gamma=0.99, 
                 tau=0.005, 
                 alpha=5.0, 
                 num_random_actions=10):
        
        # Detect GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        
        # CQL penalty coefficient
        self.alpha = alpha
        
        # Number of random actions to sample for the conservative penalty
        self.num_random_actions = num_random_actions
        
        # Initialize networks
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net).to(self.device)
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.target_policy = copy.deepcopy(self.policy).to(self.device)
        
        # Initialize optimizers
        self.q_optim = optim.Adam(self.q_net.parameters(), lr=lr)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr / 10)  # Lower policy LR (heuristic)

        # Logging
        self.q_loss = []
        self.policy_loss = []

    def get_action(self, state, deterministic=False):
        """Select action from current policy; add a bit of noise if not deterministic."""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.policy(state)

            if not deterministic:
                noise = 0.1 * torch.randn_like(action).to(self.device)
                action = torch.clamp(action + noise, -1.0, 1.0)

            return action.squeeze(0).cpu().numpy()

    def get_q_loss(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions = self.target_policy(next_states)
            next_q_values = self.target_q_net(next_states, next_actions)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        q_values = self.q_net(states, actions)
        bellman_error = F.mse_loss(q_values, target_q_values)

        batch_size = states.shape[0]

        with torch.no_grad():
          random_actions = torch.FloatTensor(batch_size, self.num_random_actions, self.action_dim).uniform_(-1, 1).to(self.device)
          current_actions = self.policy(states).unsqueeze(1).expand(-1, self.num_random_actions, -1)
          next_actions = self.policy(next_states).unsqueeze(1).expand(-1, self.num_random_actions, -1)

        repeated_states = states.unsqueeze(1).repeat(1, self.num_random_actions, 1).view(-1, self.state_dim)
        q_rand = self.q_net(repeated_states, random_actions.reshape(-1, self.action_dim)).reshape(batch_size, self.num_random_actions, 1)
        q_current = self.q_net(repeated_states, current_actions.reshape(-1, self.action_dim)).reshape(batch_size, self.num_random_actions, 1)
        q_next = self.q_net(repeated_states, next_actions.reshape(-1, self.action_dim)).reshape(batch_size, self.num_random_actions, 1)

        q_cat = torch.cat([q_rand, q_current, q_next], dim=1)
        cql_loss = (torch.logsumexp(q_cat, dim=1) - q_values).mean()

        total_q_loss = bellman_error + self.alpha * cql_loss
        return total_q_loss

    def get_policy_loss(self, states):
        """
        Simple policy loss for CQL:
          maximize Q(s, pi(s)) => minimize -Q(s, pi(s))
        """
        actions = self.policy(states)
        q_values = self.q_net(states, actions)
        policy_loss = -q_values.mean()
        return policy_loss

    def update(self, states, actions, rewards, next_states, dones):        
        # ---- Update Q-function ----
        self.q_optim.zero_grad()
        q_loss = self.get_q_loss(states, actions, rewards, next_states, dones)
        q_loss.backward()
        self.q_optim.step()
        self.q_loss.append(q_loss.item())
        
        # ---- Update Policy ----
        self.policy_optim.zero_grad()
        policy_loss = self.get_policy_loss(states)
        policy_loss.backward()
        self.policy_optim.step()
        self.policy_loss.append(policy_loss.item())
        
        # ---- Soft-update target networks ----
        for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.policy.parameters(), self.target_policy.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
