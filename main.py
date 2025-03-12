import gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import random

from replay_buffer import ReplayBuffer
from model import CQLAgent

# Hyperparameters
BATCH_SIZE = 256  # Increased batch size for more stable learning
MAX_EPISODES = 500  # Increased from 10 to allow sufficient training time
MAX_STEPS = 1000
BUFFER_SIZE = 100000
LOG_INTERVAL = 10
LEARNING_RATE = 3e-5  # Reduced learning rate for more stability

# Initialize environment
env = gym.make('InvertedPendulum-v4', render_mode=None)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Setup device for training
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize logging and agent
writer = SummaryWriter()
agent = CQLAgent(state_dim, action_dim, lr=LEARNING_RATE, alpha=5.0)  # Added explicit alpha parameter

# Move agent model components to GPU
agent.q_net.to(device)
agent.target_q_net.to(device)
agent.policy.to(device)
agent.target_policy.to(device)

replay_buffer = ReplayBuffer(BUFFER_SIZE)

# Tracking losses and metrics
policy_losses = []
q_losses = []
cql_losses = []  # Track CQL-specific losses
episode_lengths = []  # Track how long episodes last

# Training loop
for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    state = torch.FloatTensor(state).to(device)
    episode_reward = 0
    steps_in_episode = 0

    for step in range(MAX_STEPS):
        # Get action from agent
        action = agent.get_action(state.cpu().numpy())

        # Clip action to match environment limits
        action = np.clip(action, env.action_space.low, env.action_space.high)

        # Step environment
        next_state, reward, done, _, _ = env.step(action)
        next_state = torch.FloatTensor(next_state).to(device)
        
        # Store original reward for logging - no scaling
        raw_reward = reward
        
        # Store experience in replay buffer (using original reward, no scaling)
        replay_buffer.add(state.cpu().numpy(), action, raw_reward, next_state.cpu().numpy(), done)

        state = next_state
        episode_reward += raw_reward
        steps_in_episode += 1

        # Update agent if enough samples exist in replay buffer
        if len(replay_buffer) > BATCH_SIZE:
            agent.update(replay_buffer, batch_size=BATCH_SIZE)

            # Store losses for visualization
            if hasattr(agent, 'q_loss'):
                q_losses.append(agent.q_loss)
            if hasattr(agent, 'policy_loss'):
                policy_losses.append(agent.policy_loss)
            if hasattr(agent, 'cql_loss'):
                cql_losses.append(agent.cql_loss)

        if done:
            break

    # Track episode length
    episode_lengths.append(steps_in_episode)

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
    if hasattr(agent, 'q_loss') and hasattr(agent, 'policy_loss'):
        print(f"Episode {episode}: Q-Loss: {agent.q_loss:.4f}, Policy Loss: {agent.policy_loss:.4f}")

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

# Plot rewards
plt.subplot(2, 2, 2)
episode_rewards = [sum(episode_lengths[:i+1]) for i in range(len(episode_lengths))]
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Training Rewards")

# Plot episode lengths
plt.subplot(2, 2, 3)
plt.plot(episode_lengths)
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Episode Lengths")

plt.tight_layout()
plt.show()

# Cleanup
writer.close()
env.close()
