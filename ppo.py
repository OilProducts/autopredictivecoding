import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym

import cProfile
import pstats

# from gymnasium.utils.save_video import save_video

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# if torch.backends.mps.is_available():
#     device = torch.device('mps')
print(f'Using device: {device}')

# Hyperparameters
learning_rate = 5e-4  # Learning rate for optimizer
gamma = 0.99  # Discount factor for future rewards
lmbda = 0.9  # Lambda for GAE-Lambda
eps_clip = 0.1  # Clipping epsilon for PPO's loss
K_epoch = 3  # Number of epochs for optimization
T_horizon = 100000  # Horizon (batch size) for collecting data per policy update


# Policy Network
class PPO(nn.Module):
    def __init__(self, in_dim, out_dim,
                 lr=1e-4,
                 gamma=0.99,
                 lmbda=0.9,
                 eps_clip=0.1,
                 K_epoch=3,
                 T_horizon=500,
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
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

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
        # print(f's_prime: {s_prime}')
        # print(f's: {s}')

        return s, a, r, s_prime, done, old_log_pi

    def train_net(self):
        # print('Training the model')
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
        actor_loss = -torch.min(surr1, surr2).mean() * 100000

        # Value loss
        critic_loss = F.smooth_l1_loss(self.v(s).squeeze(-1), td_target.squeeze()).mean()

        # print(f"Actor loss: {actor_loss.item()}, Critic loss: {critic_loss.item()}")

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.data = []  # Clear data after training

    def act_and_train(self, state, reward, done):
        self.put_data((self.last_s,
                       self.last_a,
                       reward / 100,
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


# Main loop
def main():
    env = gym.make('LunarLander-v2', render_mode='rgb_array_list')  # Create the game environment
    model = PPO(8, 4, learning_rate, gamma, lmbda, eps_clip, K_epoch, T_horizon, device).to(device)
    total_score = 0.0
    current_best_score = -np.inf
    print_interval = 20

    start_time = time.time()
    for n_epi in range(10000):
        state, _ = env.reset()
        model.reset()  # Reset model state tracking
        state = torch.from_numpy(state).float().to(device)

        # First round, so we don't end up storing the initial state along with 0 reward
        action = model.act(state)

        state, reward, truncated, complete, _ = env.step(action)
        state = torch.from_numpy(state).float().to(device)

        done = False
        score = 0.0
        while not done:
            action = model.act_and_train(state, reward, done)
            state, reward, truncated, complete, _ = env.step(action)
            state = torch.from_numpy(state).float().to(device)
            score += reward
            done = truncated or complete
        action = model.act_and_train(state, reward, done)

        end_time = time.time()
        total_score += score
        # print(f'Episode {n_epi} finished in {end_time - start_time:.2f} seconds, score: {score:.1f}')
        if n_epi % print_interval == 0 and n_epi != 0:
            end_time = time.time()

            print(f'Episode {n_epi - print_interval} - {n_epi} finished in '
                  f'{end_time - start_time:.2f} seconds, avg score: {total_score / print_interval:.1f}')
            start_time = time.time()
            total_score = 0.0

    env.close()


def run():
    main()  # Assuming your main function includes all the important operations


if __name__ == '__main__':
    # profiler = cProfile.Profile()
    # profiler.enable()
    # run()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()
    run()
