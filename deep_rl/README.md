# 🚀 Deep Reinforcement Learning — DQN, DDQN, Dueling DQN, PER

This repository contains clean and beginner-friendly implementations of major Deep Reinforcement Learning algorithms.

## Algorithms

* DQN (Deep Q-Network)
* DDQN (Double DQN)
* Dueling DQN
* PER-DQN (Prioritized Experience Replay)

## Project Structure

```
deep_rl/
│
├── dqn/
│   ├── model.py
│   ├── replay_buffer.py
│   ├── dqn_agent.py
│   └── train_cartpole.py
│
├── ddqn/
├── dueling_dqn/
├── per_dqn/
└── README.md
```

## Training

Install requirements:

```
pip install -r requirements.txt
```

Run the baseline DQN example:

```
python deep_rl/dqn/train_cartpole.py
```

## Learning Goals

* Understand Q-learning
* Learn neural network function approximation
* Explore target networks and replay buffers
* Extend to DDQN, Dueling, PER

## License

MIT
