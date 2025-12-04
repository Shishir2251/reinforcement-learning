import numpy as np

class UCB1:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)      
        self.values = np.zeros(n_arms)      
        self.total_pulls = 0

    def select_action(self):
    
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

    
        ucb_values = self.values + np.sqrt(2 * np.log(self.total_pulls) / self.counts)

        return np.argmax(ucb_values)

    def update(self, action, reward):
        self.counts[action] += 1
        self.total_pulls += 1

        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n
