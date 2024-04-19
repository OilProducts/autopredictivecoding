import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym

# Hyperparameters
learning_rate = 0.002   # Learning rate for optimizer
gamma = 0.99            # Discount factor for future rewards
lmbda = 0.95            # Lambda for GAE-Lambda
eps_clip = 0.1          # Clipping epsilon for PPO's loss
K_epoch = 3             # Number of epochs for optimization
T_horizon = 20          # Horizon (batch size) for collecting data per policy update

# Policy Network
class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []  # This will hold samples collected for training

        # Neural network for policy
        self.fc1 = nn.Linear(4, 256)  # First fully connected layer
        self.fc_pi = nn.Linear(256, 2)  # Output layer for action probabilities
        self.fc_v = nn.Linear(256, 1)  # Output layer for state value estimation
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        # Forward pass through the network for policy
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc_pi(x)
        prob = torch.softmax(x, dim=softmax_dim)  # Use softmax to get action probabilities
        return prob

    def v(self, x):
        # Forward pass through the network for value
        x = self.fc1(x)
        x = torch.relu(x)
        v = self.fc_v(x)
        return v

    def put_data(self, transition, prob_a):
        # Include the log of probability of the action taken
        self.data.append(transition + (prob_a,))

    # def put_data(self, transition):
    #     # Store transition data for training later
    #     self.data.append(transition)

    def make_batch(self):
        # Make batches from collected data, including old log probabilities
        s_lst, a_lst, r_lst, s_prime_lst, done_lst, old_log_pi_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done, old_log_pi = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_lst.append([done])
            old_log_pi_lst.append([old_log_pi])

        s, a, r, s_prime, done, old_log_pi = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                  torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                  torch.tensor(done_lst, dtype=torch.float), torch.tensor(old_log_pi_lst, dtype=torch.float)
        self.data = []  # Clear data after processing
        return s, a, r, s_prime, done, old_log_pi

    def train_net(self):
        # Training the model using the collected batch data
        s, a, r, s_prime, done, old_log_pi = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * (1 - done)
        delta = td_target - self.v(s)  # Compute delta for advantage estimation

        advantage_lst = []
        advantage = 0.0
        for delta_t in torch.flip(delta, [0]):  # Compute advantages using GAE-lambda
            advantage = gamma * lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float)
        print(f'advantage: {advantage}, mean: {advantage.mean()}, std: {advantage.std()}')
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        new_log_pi = torch.log(pi_a)

        ratio = torch.exp(new_log_pi - old_log_pi)  # Correct ratio calculation

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
        loss = -torch.min(surr1, surr2).mean() + F.smooth_l1_loss(self.v(s).squeeze(-1), td_target.squeeze(-1)).mean()

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

# Main loop
def main():
    env = gym.make('CartPole-v1')  # Create the game environment
    model = PPO()
    score = 0.0
    print_interval = 1000

    for n_epi in range(100000):
        s, _ = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = torch.distributions.Categorical(prob)
                a = m.sample()
                prob_a = torch.log(prob.squeeze(0)[a]).item()  # Log probability of the action taken
                s_prime, r, term, trunc, info = env.step(a.item())
                done = term or trunc
                model.put_data((s, a, r / 100.0, s_prime, done), prob_a)  # Store data with log prob
                s = s_prime

                score += r
                if done:
                    break

            model.train_net()  # Train the model after collecting enough data

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()
