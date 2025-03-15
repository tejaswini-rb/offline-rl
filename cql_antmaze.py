# %%
import gymnasium as gym
import gymnasium_robotics
from gym.wrappers import RecordVideo
import minari
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

from replay_buffer import ReplayBuffer
from model import CQLAgent
from evaluate import total_rewards, record_video
import torch

# %%
BATCH_SIZE = 64
MAX_EPISODES = 1000
MAX_UPDATES = 10000  # Number of gradient updates

# Load AntMaze Offline Dataset
dataset = minari.load_dataset('D4RL/antmaze/umaze-v1', download=True)
# dataset = dataset.filter_episodes(lambda episode: episode.rewards.mean() > 2)
env  = dataset.recover_environment()
# gym.register_envs(gymnasium_robotics)

# env = gym.make('AntMaze_UMazeDense-v5', render_mode='rgb_array')
# env = gym.make('InvertedPendulum-v4', render_mode='rgb_array')
state_dim = env.observation_space['observation'].shape[0]
action_dim = env.action_space.shape[0]

# Initialize Agent and Replay Buffer
replay_buffer = ReplayBuffer(100000)
num_episodes = 100  # Number of episodes to collect
max_steps = 1000   # Max steps per episode

# for episode in range(num_episodes):
#     obs = env.reset()
#     state = obs[0]['observation']  # Extract observation
#     done = False
#     step = 0

#     while not done and step < max_steps:
#         action = env.action_space.sample()  # Random action
#         next_obs, reward, terminated, truncated, _ = env.step(action)
#         next_state = next_obs['observation']
#         done = terminated or truncated

#         # Store in replay buffer
#         replay_buffer.add(state, action, reward, next_state, done)

#         # Move to next state
#         state = next_state
#         step += 1

# print(f"Replay buffer filled with {len(replay_buffer)} transitions.")
# Fill Replay Buffer with Offline Data

dataset_episodes = dataset.sample_episodes(MAX_EPISODES)
for episode in dataset_episodes:
    episode_rewards = episode.rewards.sum()
    # print(f"Episode Reward: {episode_rewards}")
    for step in range(episode.actions.shape[0]):
        state = episode.observations['observation'][step]
        action = episode.actions[step]
        reward = episode.rewards[step] 
        next_state = episode.observations['observation'][step + 1] if step + 1 < episode.actions.shape[0] else state
        done = episode.terminations[step] or episode.truncations[step]
        
        replay_buffer.add(state, action, reward, next_state, done)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

writer = SummaryWriter()
agent = CQLAgent(state_dim, action_dim, gamma=0.7, alpha=0.2)

# Move agent model components to GPU
agent.q_net.to(device)
agent.target_q_net.to(device)
agent.policy.to(device)
agent.target_policy.to(device)
# Train CQLAgent on Offline Data
for update_step in range(MAX_UPDATES):
    
    if len(replay_buffer) > BATCH_SIZE:
        agent.update(replay_buffer, batch_size=BATCH_SIZE)

        # Logging every 1000 steps
        if update_step % 10 == 0:
            writer.add_scalar("Q loss", agent.q_loss, update_step)
            writer.add_scalar("Policy loss", agent.policy_loss, update_step)
            writer.add_scalar('CQL loss', agent.cql_loss, update_step)
            print(f"Update {update_step}: Q Loss = {agent.q_loss}, Policy Loss = {agent.policy_loss}, CQL Loss = {agent.cql_loss}")

# Plot Loss Curves
# plt.plot(agent.policy_loss, label="Policy Loss")
# plt.plot(agent.q_loss, label="Q Loss")
# plt.xlabel("Update Step")
# plt.ylabel("Loss")
# plt.title("Training Losses")
# plt.legend()
# plt.show()

# plt.plot(agent.q_values, label="Q Value")
# plt.xlabel("Update Step")
# plt.ylabel("Q Value")
# plt.title("Q Value")
# plt.show()

writer.close()
env.close()

# %%
# Evaluate the Agent
all_rewards = total_rewards(agent,env)
# %%
total_rewards = []
num_episodes = 50
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # Extract observation if needed
        state = obs['observation'] if isinstance(obs, dict) else obs[0]['observation']
        action = agent.get_action(state)
        print(action)

        obs,reward,terminated,truncated,_ = env.step(action)
        episode_reward += reward
        done = terminated or truncated

    total_rewards.append(episode_reward)
    # print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
# %% make the video
import os
from gym.wrappers import RecordVideo
from IPython.display import Video, display, clear_output
os.environ["MUJOCO_GL"] = "egl"
def visualize(agent):
    """Visualize agent with a custom camera angle."""

    # Create environment in rgb_array mode
    env = gym.make("AntMaze_UMazeDense-v5", render_mode="rgb_array")

    # Apply video recording wrapper
    env = RecordVideo(env, video_folder="./", episode_trigger=lambda x: True)

    # Access the viewer object through mujoco_py
    # viewer = env.unwrapped.mujoco_renderer.viewer  # Access viewer
    # viewer.cam.distance = 3.0     # Set camera distance
    # viewer.cam.azimuth = 90       # Rotate camera around pendulum
    # viewer.cam.elevation = 0   # Tilt the camera up/down
    # env.start_video_recorder()

    obs = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # Extract observation if needed
        state = obs['observation'] if isinstance(obs, dict) else obs[0]['observation']
        action = agent.get_action(state)

        obs,reward,terminated,truncated,_ = env.step(action)
        episode_reward += reward
        done = terminated or truncated

        # Call render to generate frames for the video
        # env.render()
    env.close()
    # Display the latest video
    clear_output(wait=True)
    display(Video("./rl-video-episode-0.mp4", embed=True))

visualize(agent)
# %%
