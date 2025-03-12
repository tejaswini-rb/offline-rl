import gym
import minari
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

from replay_buffer import ReplayBuffer
from model import CQLAgent


BATCH_SIZE = 64
MAX_EPISODES = 100
MAX_UPDATES = 1000  # Number of gradient updates
LOG_INTERVAL = 10

# Load AntMaze Offline Dataset
dataset = minari.load_dataset('D4RL/antmaze/umaze-v1', download=True)
env  = dataset.recover_environment()

# Setup device for training
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

state_dim = env.observation_space['observation'].shape[0]
action_dim = env.action_space.shape[0]

# Initialize Agent and Replay Buffer
replay_buffer = ReplayBuffer(100000)

# # Fill Replay Buffer with Offline Data
dataset_episodes = dataset.sample_episodes(MAX_EPISODES)
for episode in dataset_episodes:
    for step in range(episode.actions.shape[0]):
        state = episode.observations['observation'][step]
        action = episode.actions[step]
        reward = episode.rewards[step]
        next_state = episode.observations['observation'][step + 1] if step + 1 < episode.actions.shape[0] else state
        done = episode.terminations[step] or episode.truncations[step]
        
        replay_buffer.add(state, action, reward, next_state, done)

policy_losses = []
q_losses = []
cql_losses = []  # Track CQL-specific losses
episode_lengths = []  # Track how long episodes last

writer = SummaryWriter()
agent = CQLAgent(state_dim, action_dim, gamma=0.7, alpha=0.1)
# Train CQLAgent on Offline Data
for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    state = torch.FloatTensor(state["observation"]).to(device)
    episode_reward = 0
    steps_in_episode = 0

    for update in range(MAX_UPDATES):
         # Get action from agent
        action = agent.get_action(state.cpu().numpy())

        # Clip action to match environment limits
        action = np.clip(action, env.action_space.low, env.action_space.high)

        # Step environment
        next_state, reward, done, _, _ = env.step(action)
        next_state = torch.FloatTensor(next_state["observation"]).to(device)
        
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
