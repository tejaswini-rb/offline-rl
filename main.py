import gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import random

from replay_buffer import ReplayBuffer
from model import CQLAgent

# Hyperparameters
BATCH_SIZE = 256  
MAX_EPISODES = 500  
MAX_STEPS = 1000
BUFFER_SIZE = 100000
LOG_INTERVAL = 10
LEARNING_RATE = 3e-4  # Match the CQL model's LR

# Initialize environment
env = gym.make('InvertedPendulum-v4', render_mode=None)
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

# Training loop
for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    episode_reward = 0
    steps_in_episode = 0

    for step in range(MAX_STEPS):
        # Get action from agent
        action = agent.get_action(state, deterministic=False)

        # Clip action to match environment limits
        action = np.clip(action, env.action_space.low, env.action_space.high)

        # Step environment
        next_state, reward, done, _, info = env.step(action)
        
        # ✅ Ensure proper termination detection
        done_reason = info.get("TimeLimit.truncated", False) if info else False
        if done_reason:
            print(f"✅ Episode {episode} ended due to time limit (success).")
        elif done:
            print(f"❌ Episode {episode} ended due to failure (fall).")

        # Store experience in replay buffer
        replay_buffer.add(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        steps_in_episode += 1

        # Update agent if enough samples exist in replay buffer
        if len(replay_buffer) > BATCH_SIZE:
            agent.update(replay_buffer, batch_size=BATCH_SIZE)

            # Store losses for visualization
            q_losses.append(agent.q_loss)
            policy_losses.append(agent.policy_loss)
            cql_losses.append(agent.cql_loss)

        if done:
            break

    # Track episode length and cumulative reward
    episode_lengths.append(steps_in_episode)
    cumulative_rewards.append(episode_reward)

    # Logging to TensorBoard
    if episode % LOG_INTERVAL == 0:
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        writer.add_scalar("Length/Episode", steps_in_episode, episode)
        if q_losses:
            writer.add_scalar("Loss/Q", q_losses[-1], episode)
        if policy_losses:
            writer.add_scalar("Loss/Policy", policy_losses[-1], episode)
        if cql_losses:
            writer.add_scalar("Loss/CQL", cql_losses[-1], episode)

    print(f"Episode {episode}, Reward: {episode_reward:.2f}, Length: {steps_in_episode}")
    print(f"Q-Loss: {agent.q_loss:.4f}, Policy Loss: {agent.policy_loss:.4f}, CQL Loss: {agent.cql_loss:.4f}")

# Plot training metrics
plt.figure(figsize=(15, 10))

# Plot losses
plt.subplot(2, 2, 1)
plt.plot(q_losses, label="Q Loss")
plt.plot(policy_losses, label="Policy Loss")
if cql_losses:
    plt.plot(cql_losses, label="CQL Loss")
plt.xlabel("Update Step")
plt.ylabel("Loss")
plt.title("Training Losses")
plt.legend()

# Plot cumulative rewards
plt.subplot(2, 2, 2)
plt.plot(cumulative_rewards)
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Training Rewards")

# Plot episode lengths
plt.subplot(2, 2, 3)
plt.plot(episode_lengths)
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Episode Lengths")

# ✅ Scatter plot for Episode Length vs Reward
plt.subplot(2, 2, 4)
plt.scatter(episode_lengths, cumulative_rewards, alpha=0.5)
plt.xlabel("Episode Length")
plt.ylabel("Cumulative Reward")
plt.title("Episode Length vs Reward")

plt.tight_layout()
plt.show()

# Cleanup
writer.close()
env.close()
