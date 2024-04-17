"""
A PPO based agent that learns
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 17),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.network(state)


class DecisionUnit:
    def __init__(self, input_size, action_space):
        self.actor = Actor()
        self.critic = Critic()
        self.optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.action_space = action_space
        # Hyperparameters
        self.lr = 0.002
        self.gamma = 0.99
        self.betas = (0.9, 0.999)
        self.eps_clip = 0.2
        self.K_epochs = 4

    def ppo_update(self, states, actions, log_probs, rewards, next_states, done_flags,
                   clip_param=0.2):
        rewards = torch.tensor(rewards, dtype=torch.float32)
        old_log_probs = torch.tensor(log_probs)
        actions = torch.tensor(actions)

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            log_probs, state_values, dist_entropy = (self.actor(states),
                                                     self.critic(states),
                                                     Categorical(self.actor(states)).entropy())

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(log_probs - old_log_probs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * (
                    state_values - rewards) ** 2 - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
