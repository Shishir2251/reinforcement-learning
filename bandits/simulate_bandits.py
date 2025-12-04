
import numpy as np
import matplotlib.pyplot as plt

from epsilon_greedy import EpsilonGreedy
from ucb import UCB1
from thompson_sampling import ThompsonBernoulli


np.random.seed(0)
true_means = np.random.randn(10)


def run(agent, steps=2000):
    rewards = []

    for _ in range(steps):
        try:
            action = agent.select_action()
        except Exception as e:
            print(f"[ERROR] agent.select_action() failed: {e}")
            break

        try:
            reward = np.random.randn() + true_means[action]
        except Exception as e:
            print(f"[ERROR] reward generation failed: {e}")
            reward = 0

        try:
            agent.update(action, reward)
        except Exception as e:
            print(f"[ERROR] agent.update() failed: {e}")

        rewards.append(reward)

    return np.array(rewards)


if __name__ == "__main__":
    print("MAIN BLOCK RUNNING...")

    try:
        eg = EpsilonGreedy(10, epsilon=0.1)
        ucb = UCB1(10)
        ts = ThompsonBernoulli(10)
    except Exception as e:
        print(f"[INIT ERROR] Failed creating agents: {e}")
        raise

    agents = {
        "epsilon_greedy": eg,
        "ucb": ucb,
        "thompson": ts,
    }

    results = {}

    for name, agent in agents.items():
        print(f"Running agent: {name}")
        try:
            rewards = run(agent)
            results[name] = rewards
        except Exception as e:
            print(f"[ERROR] Failed running agent {name}: {e}")

    try:
        plt.figure(figsize=(10, 6))
        for name, r in results.items():
            avg = np.cumsum(r) / (np.arange(1, len(r) + 1))
            plt.plot(avg, label=name)

        plt.title("Bandit Algorithm Comparison")
        plt.xlabel("Steps")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"[PLOT ERROR] Failed to plot results: {e}")
