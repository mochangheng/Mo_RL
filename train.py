import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
import seaborn as sns

from replay import ReplayBuffer
from DQN import DQN
from utils import TransitionTuple
from envs import registry

START_TIMESTEP = 1000
TOTAL_TIMESTEPS = 20000
BATCH_SIZE = 100
RENDER = False
OUTPUT_PLOT = "DQN.png"
env_name = "cart_pole"

def main():
    env_spec = registry[env_name]
    env = gym.make(env_spec["id"])
    ep_max_steps = env_spec["max_episode_steps"]
    agent = DQN(env.observation_space.shape, env.action_space.n)
    replay_buffer = ReplayBuffer()

    state = env.reset()
    done = False
    ep_timesteps = 0
    ep_reward = 0
    ep_num = 0
    reward_history = []

    for t in range(TOTAL_TIMESTEPS):
        ep_timesteps += 1

        # Select action
        if t < START_TIMESTEP:
            action = env.action_space.sample()
        else:
            action = agent.select_action(np.array(state))

        # Perform action
        next_state, reward, done, _ = env.step(action)
        train_done = done and ep_timesteps < ep_max_steps

        replay_buffer.add(TransitionTuple(state, action, next_state, reward, int(train_done)))

        state = next_state
        ep_reward += reward
        
        if t >= START_TIMESTEP:
            agent.train(replay_buffer, BATCH_SIZE)

        if done:
            reward_history.append(ep_reward)
            print(f"[Episode {ep_num+1}, Timestep {t+1}] Total reward: {ep_reward}  Total timesteps: {ep_timesteps}")
            state = env.reset()
            done = False
            ep_timesteps = 0
            ep_reward = 0
            ep_num += 1

        if RENDER:
            env.render()

    # Visualize results
    if OUTPUT_PLOT:
        sns.lineplot(x=np.arange(len(reward_history))+1, y=reward_history)
        plt.ylabel("Episode Reward")
        plt.xlabel("Episode Number")
        plt.savefig(OUTPUT_PLOT)

if __name__ == "__main__":
    main()