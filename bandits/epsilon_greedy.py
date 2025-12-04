import numpy as np

class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)      # Number of times each arm was pulled
        self.values = np.zeros(n_arms)      # Estimated value of each arm

    def select_action(self):
        if np.random.rand() < self.epsilon:
            # Explore
            return np.random.randint(self.n_arms)
        else:
            # Exploit
            return np.argmax(self.values)

    def update(self, action, reward):
        self.counts[action] += 1
        n = self.counts[action]

        # Incremental mean update
        self.values[action] += (reward - self.values[action]) / n
