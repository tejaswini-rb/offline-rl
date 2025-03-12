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

# CQL Agent Definition (No SAC)
class CQLAgent:
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 lr=3e-4, 
                 gamma=0.99, 
                 tau=0.0025, 
                 alpha=3.0, 
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
        self.q_loss = 0
        self.policy_loss = 0
        self.cql_loss = 0

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
        random_actions = torch.FloatTensor(
            np.random.uniform(-1, 1, (batch_size, self.num_random_actions, self.action_dim))
        ).to(self.device)

        # Compute Q-values for random actions
        random_q_values = []
        for i in range(self.num_random_actions):
            rand_q = self.q_net(states, random_actions[:, i, :])
            random_q_values.append(rand_q)
        random_q_values = torch.cat(random_q_values, dim=1)

        # No in-place modification here:
        random_density = torch.logsumexp(random_q_values, dim=1, keepdim=True)
        random_density = random_density - np.log(self.num_random_actions)

        cql_loss = (random_density - q_values).mean()

        total_q_loss = bellman_error + self.alpha * cql_loss
        self.cql_loss = cql_loss.item()
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

    def update(self, replay_buffer, batch_size=256):
        if len(replay_buffer) < batch_size:
            return
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # ---- Update Q-function ----
        self.q_optim.zero_grad()
        q_loss = self.get_q_loss(states, actions, rewards, next_states, dones)
        q_loss.backward()
        self.q_optim.step()
        self.q_loss = q_loss.item()
        
        # ---- Update Policy ----
        self.policy_optim.zero_grad()
        policy_loss = self.get_policy_loss(states)
        policy_loss.backward()
        self.policy_optim.step()
        self.policy_loss = policy_loss.item()
        
        # ---- Soft-update target networks ----
        for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.policy.parameters(), self.target_policy.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
