import gym
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

from replay_buffer import ReplayBuffer
from model import CQLAgent

BATCH_SIZE = 64
MAX_EPISODES = 100
MAX_STEPS = 1000

env = gym.make('InvertedPendulum-v4')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

writer = SummaryWriter()

agent = CQLAgent(state_dim, action_dim)
replay_buffer = ReplayBuffer(100000)

for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    episode_reward = 0

    for step in range(MAX_STEPS):
        action = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        reward_to_go = 0.01 * reward
        replay_buffer.add(state, action, reward_to_go, next_state, done)
        state = next_state
        episode_reward += reward_to_go

        if len(replay_buffer) > BATCH_SIZE:
            agent.update(replay_buffer, batch_size=BATCH_SIZE)
        
        if done:
            break

    # writer.add_scalar("reward", episode_reward, episode)
    # writer.add_scalar("Q loss", agent.q_losses[-1], episode)
    # writer.add_scalar("Policy loss", agent.policy_losses[-1], episode)

    print(f"Episode {episode}, Reward: {episode_reward}")

plt.plot(agent.policy_losses, label="Policy Loss")
plt.plot(agent.q_losses, label="Q Loss")
plt.xlabel("Update Step")
plt.ylabel("Loss")
plt.title("Training Losses")
plt.legend()
plt.show()

writer.close()
env.close()