import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import copy

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state, action):
        return self.fc(torch.cat([state, action], dim=1))

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state):
        return self.fc(state)
    
class CQLAgent:
    def __init__(self, state_dim, action_dim, q_lr=3e-4, policy_lr=3e-5, gamma=0.99, alpha=0.2):
        self.q_net = QNetwork(state_dim, action_dim)
        self.policy = PolicyNetwork(state_dim, action_dim)

        self.target_q_net = copy.deepcopy(self.q_net)
        self.target_policy = copy.deepcopy(self.policy)
        
        self.q_optim = optim.Adam(self.q_net.parameters(), lr=q_lr)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=policy_lr)

        self.q_losses = []
        self.policy_losses = []
        self.q_values = []

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.alpha = alpha

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.policy(state).detach().numpy()[0]
    
    def get_q_loss(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions = self.target_policy(next_states)
            next_q_values = self.target_q_net(next_states, next_actions)
        
        # target_q_values = rewards + self.gamma * next_q_values * dones
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        q_values = self.q_net(states, actions)
        q_loss = F.mse_loss(q_values, target_q_values)

        # all_actions = torch.linspace(-1, 1, steps=10).repeat(len(states), 1).to(states.device)
        all_actions = torch.rand((len(states), 10, self.action_dim)).to(states.device) * 2 - 1 # continuous action space
        # q_all = self.q_net(states.repeat_interleave(10, dim=0), all_actions.view(-1, self.action_dim))
        q_all = self.q_net(states.unsqueeze(1).repeat(1, 10, 1).view(-1, self.state_dim), all_actions.view(-1, self.action_dim))
        q_all = q_all.view(len(states), 10)
        
        logsumexp_q = torch.logsumexp(q_all, dim=1, keepdim=True)
        q_data = self.q_net(states, actions) 
        cql_loss = self.alpha * (logsumexp_q.mean() - q_data.mean())  

        return q_loss + cql_loss, q_values.mean().item()

    def get_policy_loss(self, states):
        actions = self.policy(states)
        q_values = self.q_net(states, actions)
        policy_loss = -(q_values - self.alpha * (actions**2).sum(dim=1, keepdim=True)).mean()
        # policy_loss = -q_values.mean()
        
        return policy_loss
    
    def update(self, replay_buffer, batch_size=256):
        if len(replay_buffer) < batch_size:
            return
    
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_loss, q_values = self.get_q_loss(states, actions, rewards, next_states, dones)
        self.q_losses.append(q_loss.item())
        self.q_values.append(q_values)

        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        policy_loss = self.get_policy_loss(states)
        self.policy_losses.append(policy_loss.item())

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        tau = 0.005
        for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for param, target_param in zip(self.policy.parameters(), self.target_policy.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
