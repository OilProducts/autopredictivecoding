import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
# from gymnasium.utils.save_video import save_video

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
learning_rate = 5e-4  # Learning rate for optimizer
gamma = 0.98  # Discount factor for future rewards
lmbda = 0.9  # Lambda for GAE-Lambda
eps_clip = 0.2  # Clipping epsilon for PPO's loss
K_epoch = 3  # Number of epochs for optimization
T_horizon = 500  # Horizon (batch size) for collecting data per policy update


# Policy Network
class PPO(nn.Module):
    def __init__(self, in_dim, out_dim, lr, gamma, lmbda, eps_clip, K_epoch, T_horizon,
                 device='cpu'):
        super(PPO, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lr = lr
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.T_horizon = T_horizon

        self.data = []  # This will hold samples collected for training

        # Neural network for policy
        self.fc1 = nn.Linear(in_dim, 256).to(device)  # First fully connected layer
        self.fc_pi1 = nn.Linear(256, 256).to(device)  # Output layer for action probabilities
        self.fc_pi2 = nn.Linear(256, out_dim).to(device)  # Output layer for action probabilities

        # Neural network for value
        self.fc_v1 = nn.Linear(256, 256).to(device)
        self.fc_v2 = nn.Linear(256, 1).to(device)  # Output layer for state value estimation
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.pi(x)

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

        s, a, r, s_prime, done, old_log_pi = (torch.tensor(s_lst, dtype=torch.float).to(device),
                                              torch.tensor(a_lst).to(device),
                                              torch.tensor(r_lst).to(device),
                                              torch.tensor(s_prime_lst, dtype=torch.float).to(
                                                  device),
                                              torch.tensor(done_lst, dtype=torch.float).to(device),
                                              torch.tensor(old_log_pi_lst,
                                                           dtype=torch.float).to(device))
        self.data = []  # Clear data after processing
        # print(f's: {s.shape}, a: {a.shape}, r: {r.shape}, s_prime: {s_prime.shape}, done: {done.shape}, old_log_pi: {old_log_pi.shape}')
        return s, a, r, s_prime, done, old_log_pi

    def train_net(self):
        # Training the model using the collected batch data
        s, a, r, s_prime, done, old_log_pi = self.make_batch()
        print(f'reward: {r}')
        print(f's: {s.shape}, a: {a.shape}, r: {r.shape}, s_prime: {s_prime.shape}, done: {done.shape}, old_log_pi: {old_log_pi.shape}')
        td_target = r + gamma * self.v(s_prime) * (1 - done)
        print(f'td_target: {td_target}')
        delta = td_target - self.v(s)  # Compute delta for advantage estimation

        advantage_lst = []
        advantage = 0.0
        for delta_t in torch.flip(delta, [0]):  # Compute advantages using GAE-lambda
            advantage = gamma * lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)
        if len(advantage_lst) > 1:
            # print(f'advantage: {advantage}, mean: {advantage.mean()}, std: {advantage.std()}')
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        else:
            pass
            # print(f'advantage: {advantage}')

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        new_log_pi = torch.log(pi_a)

        ratio = torch.exp(new_log_pi - old_log_pi)  # Correct ratio calculation

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.smooth_l1_loss(self.v(s).squeeze(-1), td_target.squeeze(-1)).mean()
        print(f"Actor loss: {actor_loss.item()}, Critic loss: {critic_loss.item()}")
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()


# Main loop
def main():
    env = gym.make('LunarLander-v2', render_mode='rgb_array_list')  # Create the game environment
    model = PPO(8, 4, learning_rate, gamma, lmbda, eps_clip, K_epoch, T_horizon).to(device)
    score = 0.0
    current_best_score = -np.inf
    print_interval = 10

    for n_epi in range(1000):
        s, _ = env.reset()
        s = torch.from_numpy(s).float().to(device)
        done = False
        while not done:
            for t in range(T_horizon):
                prob = model(s)
                m = torch.distributions.Categorical(prob)
                a = m.sample()
                prob_a = torch.log(prob.squeeze(0)[a]).item()  # Log probability of the action taken
                s_prime, r, term, trunc, info = env.step(a.item())
                s_prime = torch.from_numpy(s_prime).float().to(device)
                done = term or trunc

                model.put_data((s.cpu().numpy(), a, r / 100.0, s_prime.cpu().numpy(), done),
                               prob_a)  #
                # Store data
                # with
                # log prob
                s = s_prime

                score += r
                if done:
                    # print(f'Episode {n_epi} finished after {t + 1} timesteps, score: {score:.1f}')
                    if score > current_best_score:
                        print(f'Saving video with score {score:.1f}')
                        # save_video(env.render(),
                        #            "videos",
                        #            episode_trigger=lambda episode_id: True,
                        #            fps=env.metadata['render_fps'],
                        #            step_starting_index=0,
                        #            episode_index=n_epi,
                        #            name_prefix=f"ppo-lunarlander-{score:.1f}", )
                        current_best_score = score
                    break

            model.train_net()  # Train the model after collecting enough data

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0

    # save_video(env.render(),
    #            "videos",
    #            fps=env.metadata['render_fps'],
    #            step_starting_index=0,
    #            episode_index=n_epi, )

    env.close()


if __name__ == '__main__':
    main()