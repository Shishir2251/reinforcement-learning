import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import DQN

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Policy network (learning network)
        self.policy_net = DQN(state_dim, action_dim).to(self.device)

        # Target network (stabilizes training)
        self.target_net = DQN(state_dim, action_dim).to(self.device)

        # ✅ Correct load_state_dict
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.gamma = gamma
        self.action_dim = action_dim

        # epsilon greedy
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.policy_net(state)
        return torch.argmax(q_values).item()

    def train_step(self, replay_buffer, batch_size=32):
        if len(replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.policy_net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        next_q_values = self.target_net(next_states)
        next_q_value = next_q_values.max(1)[0]

        target = rewards + self.gamma * next_q_value * (1 - dones)

        loss = nn.MSELoss()(q_value, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
