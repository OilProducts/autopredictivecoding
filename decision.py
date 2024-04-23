import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np


class PPO(nn.Module):
    def __init__(self, in_dim, out_dim,
                 lr=1e-4,
                 gamma=0.9999,
                 lmbda=0.9,
                 eps_clip=0.1,
                 K_epoch=3,
                 T_horizon=1000,
                 device='cpu'):
        super(PPO, self).__init__()
        self.last_log_prob = None
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.T_horizon = T_horizon

        self.last_s = None
        self.last_a = None
        self.data = []  # This will hold samples collected for training

        # Neural network for policy
        self.fc1 = nn.Linear(in_dim, 256).to(self.device)  # First fully connected layer
        self.fc_pi1 = nn.Linear(256, 256).to(self.device)
        self.fc_pi2 = nn.Linear(256, out_dim).to(
            self.device)  # Output layer for action probabilities

        # Neural network for value
        self.fc_v1 = nn.Linear(256, 256).to(self.device)
        self.fc_v2 = nn.Linear(256, 1).to(self.device)  # Output layer for state value estimation
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        return self.pi(x), self.v(x)

    def pi(self, x, softmax_dim=0):
        # Forward pass through the network for policy
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc_pi1(x)
        x = torch.relu(x)
        x = self.fc_pi2(x)
        prob = torch.softmax(x, dim=softmax_dim)  # Use softmax to get action probabilities
        return prob

    def v(self, x):
        # Forward pass through the network for value
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc_v1(x)
        x = torch.relu(x)
        v = self.fc_v2(x)
        return v

    def put_data(self, transition):
        # Include the log of probability of the action taken
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst, old_log_pi_lst = zip(*self.data)
        s, a, r, s_prime, done, old_log_pi = (
            torch.tensor(np.array(s_lst), dtype=torch.float).to(self.device),
            torch.tensor(np.array(a_lst), dtype=torch.long).to(self.device),
            torch.tensor(np.array(r_lst), dtype=torch.float).to(self.device),
            torch.tensor(np.array(s_prime_lst), dtype=torch.float).to(self.device),
            torch.tensor(np.array(done_lst), dtype=torch.float).to(self.device),
            torch.tensor(np.array(old_log_pi_lst), dtype=torch.float).to(self.device))
        self.data = []  # Clear data after processing

        return s, a, r, s_prime, done, old_log_pi

    def train_net(self):
        # Training the model using the collected batch data
        s, a, r, s_prime, done, old_log_pi = self.make_batch()
        td_target = r + self.gamma * self.v(s_prime).squeeze(-1) * (1 - done)
        delta = td_target - self.v(s)  # Compute delta for advantage estimation

        # Compute advantages using GAE-lambda
        advantage_lst = [0] * len(delta)
        advantage = 0.0
        for i in reversed(range(len(delta))):
            advantage = self.gamma * self.lmbda * advantage + delta[i]
            advantage_lst[i] = advantage

        if len(advantage_lst) > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        else:
            pass

        # Policy loss
        pi = self.pi(s, softmax_dim=1)
        a = a.unsqueeze(1)
        current_log_pi = torch.log(pi.gather(1, a))
        ratio = torch.exp(current_log_pi - old_log_pi.unsqueeze(1))
        surr1 = ratio * advantage.unsqueeze(1)
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage.unsqueeze(1)
        actor_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        critic_loss = F.smooth_l1_loss(self.v(s).squeeze(-1), td_target.squeeze()).mean()

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.data = []  # Clear data after training

    def act_and_train(self, state, reward, done):
        self.put_data((self.last_s,
                       self.last_a,
                       reward,
                       state.cpu().numpy(),
                       done,
                       self.last_log_prob))

        if done:
            self.train_net()
            self.reset()
            return

        action = self.act(state)

        if len(self.data) >= self.T_horizon:
            self.train_net()
            self.reset(soft=True)

        return action

    def act(self, state):
        # Choose the first action in the episode
        prob, value = self(state)
        dist = torch.distributions.Categorical(prob)
        action = dist.sample()
        self.last_s = state.cpu().numpy()
        self.last_a = action.item()
        self.last_log_prob = torch.log(prob.squeeze(0)[action]).item()
        return action.item()

    def reset(self, soft=False):
        # Call this method when the environment is reset (i.e., at the start of each new episode)
        if not soft:
            self.last_s = None
            self.last_a = None
            self.last_log_prob = None
        else:
            print('Soft reset')
        self.data = []


class Actor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        prob = F.softmax(self.fc3(x), dim=-1)
        return prob


class Critic(nn.Module):
    def __init__(self, in_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class PPOSplit(nn.Module):
    def __init__(self, in_dim, out_dim, lr=1e-4, gamma=0.98, lmbda=0.95, eps_clip=0.1, K_epoch=3,
                 T_horizon=1000, device='cpu'):
        super(PPOSplit, self).__init__()
        self.device = device
        self.actor = Actor(in_dim, out_dim).to(self.device)
        self.critic = Critic(in_dim).to(self.device)

        self.optimizerA = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizerC = optim.Adam(self.critic.parameters(), lr=lr)

        self.last_log_prob = None
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.T_horizon = T_horizon

        self.last_s = None
        self.last_a = None
        self.data = []  # This will hold samples collected for training
        # rest of the initialization remains the same

    def put_data(self, transition):
        # Include the log of probability of the action taken
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst, old_log_pi_lst = zip(*self.data)
        s, a, r, s_prime, done, old_log_pi = (
            torch.tensor(np.array(s_lst), dtype=torch.float).to(self.device),
            torch.tensor(np.array(a_lst), dtype=torch.long).to(self.device),
            torch.tensor(np.array(r_lst), dtype=torch.float).to(self.device),
            torch.tensor(np.array(s_prime_lst), dtype=torch.float).to(self.device),
            torch.tensor(np.array(done_lst), dtype=torch.float).to(self.device),
            torch.tensor(np.array(old_log_pi_lst), dtype=torch.float).to(self.device))
        self.data = []  # Clear data after processing

        return s, a, r, s_prime, done, old_log_pi

    def forward(self, x):
        prob = self.actor(x)
        value = self.critic(x)
        return prob, value

    # Modify the train_net method to update actor and critic using their own optimizers
    def train_net(self):
        # Prepare data and compute td_target and delta without any gradient tracking
        s, a, r, s_prime, done, old_log_pi = self.make_batch()
        with torch.no_grad():
            td_target = r + self.gamma * self.critic(s_prime).squeeze(-1) * (1 - done)
            delta = td_target - self.critic(s)

        # Compute advantages using GAE-lambda
        advantage_lst = [0] * len(delta)
        advantage = 0.0
        for i in reversed(range(len(delta))):
            advantage = self.gamma * self.lmbda * advantage + delta[i]
            advantage_lst[i] = advantage

        if len(advantage_lst) > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        else:
            pass

        # Policy loss
        pi = self.actor(s)
        a = a.unsqueeze(1)
        current_log_pi = torch.log(pi.gather(1, a))
        ratio = torch.exp(current_log_pi - old_log_pi.unsqueeze(1))
        surr1 = ratio * advantage.unsqueeze(1)
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage.unsqueeze(1)
        actor_loss = -torch.min(surr1, surr2).mean() * 100000

        # Value loss
        critic_loss = F.smooth_l1_loss(self.critic(s).squeeze(-1), td_target.squeeze()).mean()

        # Update actor
        self.optimizerA.zero_grad()
        actor_loss.backward()
        self.optimizerA.step()

        # Update critic
        self.optimizerC.zero_grad()
        critic_loss.backward()
        self.optimizerC.step()
        self.data = []  # Clear data after training

    def act_and_train(self, state, reward, done):
        self.put_data((self.last_s,
                       self.last_a,
                       reward,
                       state.cpu().numpy(),
                       done,
                       self.last_log_prob))

        if done:
            self.train_net()
            self.reset()
            return

        action = self.act(state)

        if len(self.data) >= self.T_horizon:
            self.train_net()
            self.reset(soft=True)

        return action

    def act(self, state):
        # Choose the first action in the episode
        prob, value = self(state)
        dist = torch.distributions.Categorical(prob)
        action = dist.sample()
        self.last_s = state.cpu().numpy()
        self.last_a = action.item()
        self.last_log_prob = torch.log(prob.squeeze(0)[action]).item()
        return action.item()

    def reset(self, soft=False):
        # Call this method when the environment is reset (i.e., at the start of each new episode)
        if not soft:
            self.last_s = None
            self.last_a = None
            self.last_log_prob = None
        else:
            print('Soft reset')
        self.data = []
