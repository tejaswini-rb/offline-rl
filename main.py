import gymnasium as gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import random

from replay_buffer import ReplayBuffer
from model import CQLAgent

# Hyperparameters
BATCH_SIZE = 256  
MAX_EPISODES = 100  
MAX_STEPS = 1000
BUFFER_SIZE = 100000
LOG_INTERVAL = 10
REWARD_MULTIPLIER = 1
LEARNING_RATE = 3e-4  # Match the CQL model's LR

# Initialize environment
env = gym.make("HalfCheetah-v5")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Setup device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize logging and agent
writer = SummaryWriter()
agent = CQLAgent(state_dim, action_dim, lr=LEARNING_RATE, alpha=3.0)

# Move agent model components to GPU
agent.q_net.to(device)
agent.target_q_net.to(device)
agent.policy.to(device)
agent.target_policy.to(device)

# Initialize replay buffer
replay_buffer = ReplayBuffer(BUFFER_SIZE)  # No device needed in the replay buffer

# Tracking losses and metrics
policy_losses = []
q_losses = []
cql_losses = []
episode_lengths = []  
cumulative_rewards = []  

for _ in range(100):
    state = env.reset()[0]
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        reward_to_go = reward * REWARD_MULTIPLIER
        done = terminated or truncated
        replay_buffer.add(state, action, reward_to_go, next_state, done)
        state = next_state

# Training loop
for episode in range(MAX_EPISODES):
    episode_reward = 0
    
    if len(replay_buffer) > BATCH_SIZE:
        batch = replay_buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = batch
        agent.update(states, actions, rewards, next_states, dones)
        episode_reward = rewards.sum().item()
        mean_q_loss = np.mean(agent.q_loss)
        mean_policy_loss = np.mean(agent.policy_loss)

        q_losses.append(mean_q_loss)
        policy_losses.append(mean_policy_loss)
        
        agent.q_loss.clear()
        agent.policy_loss.clear()

        print(f"Episode {episode}, Reward: {episode_reward:.2f}")
        print(f"Q-Loss: {mean_q_loss:.4f}, Policy Loss: {mean_policy_loss:.4f}")

    # Track episode length and cumulative reward
    cumulative_rewards.append(episode_reward)

    # Logging to TensorBoard
    if episode % LOG_INTERVAL == 0:
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        if q_losses:
            writer.add_scalar("Loss/Q", q_losses[-1], episode)
        if policy_losses:
            writer.add_scalar("Loss/Policy", policy_losses[-1], episode)
        if cql_losses:
            writer.add_scalar("Loss/CQL", cql_losses[-1], episode)

# Plot training metrics
plt.figure(figsize=(15, 10))

# Plot losses
plt.subplot(1, 2, 1)
plt.plot(q_losses, label="Q Loss")
plt.plot(policy_losses, label="Policy Loss")
if cql_losses:
    plt.plot(cql_losses, label="CQL Loss")
plt.xlabel("Update Step")
plt.ylabel("Loss")
plt.title("Training Losses")
plt.legend()

# Plot cumulative rewards
plt.subplot(1, 2, 2)
plt.plot(cumulative_rewards)
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Training Rewards")

'''
# Plot episode lengths
plt.subplot(2, 2, 3)
plt.plot(episode_lengths)
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Episode Lengths")

# Scatter plot for Episode Length vs Reward
plt.subplot(2, 2, 4)
plt.scatter(episode_lengths, cumulative_rewards, alpha=0.5)
plt.xlabel("Episode Length")
plt.ylabel("Cumulative Reward")
plt.title("Episode Length vs Reward")
'''

plt.tight_layout()
plt.show()

# Cleanup
writer.close()
env.close()
