# %%
import gym
import minari
# import d4rl  # Import d4rl for offline RL environments
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

from replay_buffer import ReplayBuffer
from model import CQLAgent

# %%
BATCH_SIZE = 64
MAX_EPISODES = 100
MAX_UPDATES = 100000  # Number of gradient updates

# Load AntMaze Offline Dataset
dataset = minari.load_dataset('D4RL/antmaze/umaze-v1', download=True)
env  = dataset.recover_environment()

state_dim = env.observation_space['observation'].shape[0]
action_dim = env.action_space.shape[0]

# Initialize Agent and Replay Buffer
writer = SummaryWriter()
agent = CQLAgent(state_dim, action_dim)
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

# Train CQLAgent on Offline Data
for update_step in range(MAX_UPDATES):
    if len(replay_buffer) > BATCH_SIZE:
        agent.update(replay_buffer, batch_size=BATCH_SIZE)

        # Logging every 1000 steps
        if update_step % 1000 == 0:
            writer.add_scalar("Q loss", agent.q_losses[-1], update_step)
            writer.add_scalar("Policy loss", agent.policy_losses[-1], update_step)
            print(f"Update {update_step}: Q Loss = {agent.q_losses[-1]}, Policy Loss = {agent.policy_losses[-1]}")

# Plot Loss Curves
plt.plot(agent.policy_losses, label="Policy Loss")
plt.plot(agent.q_losses, label="Q Loss")
plt.xlabel("Update Step")
plt.ylabel("Loss")
plt.title("Training Losses")
plt.legend()
plt.show()

writer.close()
env.close()

# %%
