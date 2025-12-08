import numpy as np
from collections import defaultdict
import random

from sympy import Q

def epsilon_greedy_policy(Q, State, nA, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(nA)
    else:
        return int(np,np.argmax(Q[State]))
    
def generate_episode(env, policy, max_step=1000):
    episode = []
    state = env.reset()
    for _ in range(max_step):
        action = policy(state)
        next_state , rewrad, done = env.step(action)
        episode.append((state, action, rewrad))
        state = next_state
        if done:
            break
    return episode

def mc_control_frist_visit(env, num_episodes=5000, gamma=1.0, epsilon=0.1):
    nA = env.n_actions
    Q = defaultdict(lambda: np.zeros(nA))
    returns_count = defaultdict(lambda: np.zeros(nA))

    for i in range(num_episodes):
        policy = lambda s: epsilon_greedy_policy(Q, s, nA, epsilon)
        episode = generate_episode(env, policy)
        G = 0.0
        visited = set()

        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if (state, action) not in visited:
                returns_sum [state][action] += G
                returns_count[state][action] += 1
                Q[state][action] = returns_sum[state][action]/ returns_count[state][action]
                visited.add((state,action))
        if i % 1000 == 0 and i >0:
            epsilon = max(0.01, epsilon * 0.9)
    policy = {s: int(np.argmax(a)) for s, a in Q.items()}
    return Q, policy

 

    