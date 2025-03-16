import gymnasium as gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import random

from replay_buffer import ReplayBuffer
from model import CQLAgent

## Hyperparameters
BATCH_SIZE = 256
MAX_EPISODES = 100
MAX_STEPS = 1000
BUFFER_SIZE = 1000000
NUM_TRAJS = 5
MAX_TRAJ_LENGTH = 1000
LOG_INTERVAL = 10
REWARD_MULTIPLIER = 1
ALPHA = 0.0
CQL_WEIGHT = 0.0
LEARNING_RATE = 3e-4  # Match the CQL model's LR

# Initialize environment
env = gym.make("InvertedPendulum-v5")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Initialize replay buffer
replay_buffer = ReplayBuffer(BUFFER_SIZE)  # No device needed in the replay buffer

for episode in range(10000):
    state, _ = env.reset()
    while True:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    print(f"Finished episode {episode}")

# Setup device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize logging and agent
writer = SummaryWriter()
agent = CQLAgent(state_dim, action_dim, lr=LEARNING_RATE, alpha=ALPHA, cql_weight=CQL_WEIGHT)

# Move agent model components to GPU
# agent.q1.to(device)
# agent.target_q_net.to(device)
# agent.policy.to(device)
# agent.target_policy.to(device)

# Tracking losses and metrics
policy_losses = []
q_losses = []
cql_losses = []
episode_lengths = []
cumulative_rewards = []

# Training loop
for episode in range(MAX_EPISODES):
    # train
    for step in range(MAX_STEPS):
        if len(replay_buffer) > BATCH_SIZE:
            batch = replay_buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = batch
            agent.update(states, actions, rewards, next_states, dones)
            mean_q_loss = np.mean(agent.q_loss)
            mean_policy_loss = np.mean(agent.policy_loss)

            q_losses.append(agent.q_loss)
            policy_losses.append(agent.policy_loss)

    # eval
    traj_reward = []
    for _ in range(NUM_TRAJS):
      state, _ = env.reset()
      episode_reward = 0
      for step in range(MAX_TRAJ_LENGTH):
          action = agent.get_action(state, deterministic=True)
          next_state, reward, terminated, truncated, _ = env.step(action)
          done = terminated or truncated
          state = next_state
          episode_reward += reward

          if done:
              break
      traj_reward.append(episode_reward)

    mean_traj_reward = np.mean(traj_reward)
    print(f"Episode: {episode}, Reward: {mean_traj_reward}")
    print(f"Q-Loss: {agent.q_loss:.4f}, Policy Loss: {agent.policy_loss:.4f}")

    cumulative_rewards.append(mean_traj_reward)

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

plt.tight_layout()
plt.show()

# Cleanup
writer.close()
env.close()