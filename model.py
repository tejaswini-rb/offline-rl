import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np

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
        return 1.0 * torch.tanh(self.fc(state))

# CQL Agent (twin Q-networks, random+policy actions in CQL penalty)
class CQLAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=3.0,
        num_random_actions=10
    ):
        # Detect GPU or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # CQL penalty coefficient
        self.alpha = alpha

        # Number of random actions for the conservative penalty
        self.num_random_actions = num_random_actions

        # -------------------------------
        #  Create twin Q networks
        # -------------------------------
        self.q1 = QNetwork(state_dim, action_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q1 = copy.deepcopy(self.q1).to(self.device)
        self.target_q2 = copy.deepcopy(self.q2).to(self.device)

        # For backward compatibility with main.py calls:
        self.q_net = self.q1
        self.target_q_net = self.target_q1

        # Create policy network
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.target_policy = copy.deepcopy(self.policy).to(self.device)

        # Initialize optimizers
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=lr)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr / 10)

        # Logging
        self.q_loss = 0.0
        self.policy_loss = 0.0
        self.cql_loss = 0.0  # We'll store sum of CQL penalties from Q1 and Q2

    def get_action(self, state, deterministic=False):
        """Select action from current policy; add noise if not deterministic."""
        with torch.no_grad():
            # Convert to Tensor on self.device
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_t = self.policy(state_t)

            if not deterministic:
                noise = 0.1 * torch.randn_like(action_t).to(self.device)
                action_t = torch.clamp(action_t + noise, -1.0, 1.0)

            # Return numpy array on CPU
            return action_t.squeeze(0).cpu().numpy()

    def get_q_loss(self, states, actions, rewards, next_states, dones):
        """
        Q loss for both networks:
          L_Q = MSE(Q1 - target) + MSE(Q2 - target)
                + alpha * (CQL_penalty1 + CQL_penalty2).

        Each "CQL_penalty" is:
          E[logsumexp(Q(s,a)) - Q(s,a_in_batch)],
        sampling random + policy actions for the logsumexp.
        """
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # -------------------------------------------
        # Standard Bellman backups
        # -------------------------------------------
        with torch.no_grad():
            next_actions = self.target_policy(next_states)
            next_q1 = self.target_q1(next_states, next_actions)
            next_q2 = self.target_q2(next_states, next_actions)
            # Use min for the target
            target_q = rewards + self.gamma * torch.min(next_q1, next_q2) * (1 - dones)

        q1_vals = self.q1(states, actions)
        q2_vals = self.q2(states, actions)

        bellman_loss = F.mse_loss(q1_vals, target_q) + F.mse_loss(q2_vals, target_q)

        # -------------------------------------------
        # CQL: sample random + policy actions
        # -------------------------------------------
        batch_size = states.shape[0]

        # 1) Random actions in [-1, 1]
        random_actions = torch.FloatTensor(
            np.random.uniform(-1, 1, (batch_size, self.num_random_actions, self.action_dim))
        ).to(self.device)

        # 2) Current policy actions
        with torch.no_grad():
            policy_actions = self.policy(states)  # shape (B, action_dim)

        # Combine random + policy => shape (B, N+1, a_dim)
        policy_actions = policy_actions.unsqueeze(1)
        all_actions = torch.cat([random_actions, policy_actions], dim=1)


        q1_vals_all = []
        q2_vals_all = []
        for i in range(all_actions.shape[1]):
            q1_i = self.q1(states, all_actions[:, i, :])
            q2_i = self.q2(states, all_actions[:, i, :])
            q1_vals_all.append(q1_i)
            q2_vals_all.append(q2_i)

        q1_vals_all = torch.cat(q1_vals_all, dim=1)  # shape (B, N+1)
        q2_vals_all = torch.cat(q2_vals_all, dim=1)  # shape (B, N+1)

        # logsumexp across those actions (then subtract log(# actions))
        logsumexp_q1 = torch.logsumexp(q1_vals_all, dim=1, keepdim=True) - np.log(q1_vals_all.shape[1])
        logsumexp_q2 = torch.logsumexp(q2_vals_all, dim=1, keepdim=True) - np.log(q2_vals_all.shape[1])

        # CQL penalty for Q1 and Q2
        cql_penalty1 = (logsumexp_q1 - q1_vals).mean()
        cql_penalty2 = (logsumexp_q2 - q2_vals).mean()
        cql_penalty = cql_penalty1 + cql_penalty2

        total_q_loss = bellman_loss + self.alpha * cql_penalty

        # Logging
        self.cql_loss = cql_penalty.item()
        return total_q_loss

    def get_policy_loss(self, states):
        """Policy update: minimize -E[min(Q1, Q2)]."""
        states = states.to(self.device)
        actions_pi = self.policy(states)
        q1_pi = self.q1(states, actions_pi)
        q2_pi = self.q2(states, actions_pi)
        q_min = torch.min(q1_pi, q2_pi)
        policy_loss = -q_min.mean()
        return policy_loss

    def update(self, replay_buffer, batch_size=256):
        """Sample from replay buffer and update Q networks + policy."""
        if len(replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # 1) Update Q-functions
        self.q1_optim.zero_grad()
        self.q2_optim.zero_grad()
        q_loss = self.get_q_loss(states, actions, rewards, next_states, dones)
        q_loss.backward()
        self.q1_optim.step()
        self.q2_optim.step()
        self.q_loss = q_loss.item()

        # 2) Update Policy
        self.policy_optim.zero_grad()
        policy_loss = self.get_policy_loss(states)
        policy_loss.backward()
        self.policy_optim.step()
        self.policy_loss = policy_loss.item()

        # 3) Soft-update target networks
        with torch.no_grad():
            for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.policy.parameters(), self.target_policy.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
