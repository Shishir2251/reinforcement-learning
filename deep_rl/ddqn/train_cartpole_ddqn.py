import numpy as np
import sys
import os
import gymnasium as gym
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dqn.replay_buffer import ReplayBuffer
from ddqn_agent import DDQNAgent


env = gym.make("CartPole-v1")

state_dim= env.observation_space.shape[0]
action_dim= env.action_space.n

agent= DDQNAgent(state_dim, action_dim)
buffer = ReplayBuffer()

episodes=300
batch_size=32

for ep in range(episodes):
    state,_=env.reset()
    total_reward= 0
    while True:
        action=agent.select_action(state)

        next_state, reward, terminated, truncated,_=env.step(action)
        done = terminated or truncated

        buffer.push(state,action, reward, next_state, done)

        agent.train_step(buffer, batch_size)

        state = next_state
        total_reward += reward

        if done:
            agent.update_target()
            break
    print(f"Episode {ep} - Reward: {total_reward}")

               