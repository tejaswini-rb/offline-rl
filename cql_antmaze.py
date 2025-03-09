# %%
import gym
import d4rl  # Import d4rl for offline RL environments
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

from replay_buffer import ReplayBuffer
from model import CQLAgent

BATCH_SIZE = 64
MAX_UPDATES = 100000  # Number of gradient updates

# Load AntMaze Offline Dataset
env = gym.make("antmaze-umaze-v2")  # Choose a different size if needed
dataset = d4rl.qlearning_dataset(env)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Initialize Agent and Replay Buffer
writer = SummaryWriter()
agent = CQLAgent(state_dim, action_dim)
replay_buffer = ReplayBuffer(len(dataset["observations"]))

# Fill Replay Buffer with Offline Data
for i in range(len(dataset["observations"])):
    state = dataset["observations"][i]
    action = dataset["actions"][i]
    reward = dataset["rewards"][i]
    next_state = dataset["next_observations"][i]
    done = dataset["terminals"][i]
    
    replay_buffer.add(state, action, reward, next_state, done)

print(f"Loaded {len(replay_buffer)} transitions into replay buffer.")

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
