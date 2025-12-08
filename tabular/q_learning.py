import numpy as np
from collections import defaultdict

def epsilon_greedy_action(Q, state, nA, epsilon):
    if np.random.rand() < epsilon:
        return np.random.radint(nA)
    else:
        return int(np.argmax(Q[state]))

def q_learning(env, num_episodes=2000,alpha=0.1, gamma=1.0, epsilon= 0.1):
    nA = env.n_actions
    Q = defaultdict(lambda: np.zeros(nA))

    for i in range(num_episodes):
        state = env.reset()
        done = False 

        while not done:
            action = epsilon_greedy_action(Q, state, nA, epsilon)
            next_state, reward, done = env.step(action)

            best_next = np.max (Q[next_state])
            td_target = reward + gamma * best_next * (not done)
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            state = next_state
        if (i+1) % 500 == 0:
            epsilon = max(0.01, epsilon * 0.9)
        policy = {s:int(np.argmax(a)) for s, a in Q.items()}
        return Q, policy