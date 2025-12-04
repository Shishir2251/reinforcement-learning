import numpy as np

class ThompsonBernoulli:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alphas = np.ones(n_arms)   # Success counts
        self.betas = np.ones(n_arms)    # Failure counts

    def select_action(self):
        # Sample from Beta(alpha, beta)
        samples = np.random.beta(self.alphas, self.betas)
        return np.argmax(samples)

    def update(self, action, reward):
        # reward must be 0 or 1
        if reward == 1:
            self.alphas[action] += 1
        else:
            self.betas[action] += 1
