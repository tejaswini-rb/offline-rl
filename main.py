import gymnasium as gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import random

from replay_buffer import ReplayBuffer
from model import CQLAgent

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)

def evaluate_policy(agent, env, num_eval_episodes=5):
    """
    Runs 'num_eval_episodes' episodes in the given 'env' using the agent's
    policy. Returns the average total reward across these episodes.
    """
    returns = []
    for ep in range(num_eval_episodes):
        # Reset environment
        state, _ = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            # Get the action from the agent's policy, in deterministic mode
            action = agent.get_action(state, deterministic=True)

            # Step the environment with that action
            next_state, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
            state = next_state

        returns.append(ep_return)
    return np.mean(returns)

BATCH_SIZE = 128
ONLINE_EPISODES = 250   # how many episodes to collect online
MAX_STEPS = 1000
BUFFER_SIZE = 100000
LOG_INTERVAL = 10
REWARD_MULTIPLIER = 1
LEARNING_RATE = 3e-4  # Match the CQL model's LR
RANDOM_EPISODES = 30  # how many random episodes to add
OFFLINE_TRAIN_STEPS = 1000  # how many offline training iterations

# Initialize environment
env = gym.make("InvertedPendulum-v5", max_episode_steps=MAX_STEPS)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# TensorBoard logging
writer = SummaryWriter()

# Initialize CQL agent and replay buffer
agent = CQLAgent(state_dim, action_dim, lr=LEARNING_RATE, alpha=1.0)
replay_buffer = ReplayBuffer(BUFFER_SIZE)

# Tracking losses and metrics
policy_losses = []
q_losses = []
cql_losses = []
cumulative_rewards = []

# ------------------------------------------------------------------------------
# (A) Short Online Training Phase
# ------------------------------------------------------------------------------
print("=== Collecting Online Episodes & Updating Agent Online ===")
evaluation_returns = []

# Online Training Loop
for ep in range(ONLINE_EPISODES):
    state, _ = env.reset()
    done = False
    ep_return = 0.0
    step_count = 0

    while not done and step_count < MAX_STEPS:
        action = agent.get_action(state, deterministic=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_return += reward
        step_count += 1

        # Store in replay buffer
        replay_buffer.add(state, action, reward * REWARD_MULTIPLIER, next_state, float(done))
        state = next_state

        # Online training update
        agent.update(replay_buffer, BATCH_SIZE)

    # Evaluate every few episodes
    if ep % 2 == 0:
        eval_ret = evaluate_policy(agent, env, num_eval_episodes=2)
        evaluation_returns.append(eval_ret)  # Track evaluation returns
        print(f"Online Episode {ep}, Return={ep_return:.1f}, Eval={eval_ret:.1f}")

# ------------------------------------------------------------------------------
# (B) Add random data to the buffer (optional)
# ------------------------------------------------------------------------------
print("\n=== Collecting Random Episodes ===")
for _ in range(RANDOM_EPISODES):
    state, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.add(
            state,
            action,
            reward * REWARD_MULTIPLIER,
            next_state,
            float(done)
        )
        state = next_state

print(f"Replay buffer size after random data: {len(replay_buffer)}")

# ------------------------------------------------------------------------------
# (C) Offline Training Loop (NO new environment interaction)
# ------------------------------------------------------------------------------
print("\n=== Offline Training Only ===")
# Offline Training Loop
for step in range(OFFLINE_TRAIN_STEPS):
    agent.update(replay_buffer, BATCH_SIZE)

    # Track batch reward sum
    if len(replay_buffer) >= BATCH_SIZE:
        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
        batch_reward_sum = rewards.sum().item()
    else:
        batch_reward_sum = 0.0

    q_losses.append(agent.q_loss)
    policy_losses.append(agent.policy_loss)
    cql_losses.append(agent.cql_loss)
    cumulative_rewards.append(batch_reward_sum)

    # Evaluate agent performance every LOG_INTERVAL steps
    if step % LOG_INTERVAL == 0:
        eval_ret = evaluate_policy(agent, env, num_eval_episodes=5)
        evaluation_returns.append(eval_ret)  # Track evaluation returns
        print(f"Offline Step {step}, Eval Return: {eval_ret:.2f}")

    # Logging to TensorBoard
    writer.add_scalar("Reward/BatchSum", batch_reward_sum, step)
    writer.add_scalar("Loss/Q", agent.q_loss, step)
    writer.add_scalar("Loss/Policy", agent.policy_loss, step)
    writer.add_scalar("Loss/CQL", agent.cql_loss, step)

# ------------------------------------------------------------------------------
# Final Evaluation
# ------------------------------------------------------------------------------
eval_episodes = 5
avg_return = evaluate_policy(agent, env, num_eval_episodes=eval_episodes)
print(f"\nEvaluated policy over {eval_episodes} episodes. Average return: {avg_return:.2f}")

# Plot training metrics
plt.figure(figsize=(15, 10))

# Plot losses
plt.subplot(1, 2, 1)
plt.plot(q_losses, label="Q Loss")
plt.plot(policy_losses, label="Policy Loss")
if cql_losses:
    plt.plot(cql_losses, label="CQL Loss")
plt.xlabel("Offline Update Step")
plt.ylabel("Loss")
plt.title("Training Losses")
plt.legend()

# Plot evaluation returns over time
plt.figure(figsize=(12, 6))
plt.plot(evaluation_returns, label="Evaluation Return")
plt.xlabel("Evaluation Step (Online + Offline)")
plt.ylabel("Average Return")
plt.title("Evaluation Return Over Time")
plt.legend()
plt.grid()
plt.show()

plt.tight_layout()
plt.show()

# Cleanup
writer.close()
env.close()