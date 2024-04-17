import argparse

import torch
import tqdm

import gymnasium as gym
import crafter

import model

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_noreward-random/0')
parser.add_argument('--steps', type=float, default=1e6)
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()

seed = args.seed

#
# env = gym.make('CrafterReward-v1')  # Or CrafterNoReward-v1
# env = crafter.Recorder(
#   env, './logs',
#   save_stats=True,
#   save_video=False,
#   save_episode=False,
# )

env = crafter.Env()
env = crafter.Recorder(
    env, args.outdir,
    save_stats=True,
    save_episode=False,
    save_video=False,
)
action_space = env.action_space

model = model.NeuronUnit()

done = True
step = 0
bar = tqdm.tqdm(total=args.steps, smoothing=0)
observation, _, terminated, truncated, _ = env.step(action_space.sample())

n_episodes = 500

for episode in range(n_episodes):
    state = env.reset()
    done = False
    while not done:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action_probs = actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        next_state, reward, done, _ = env.step(action.cpu().numpy())
        actor_loss, critic_loss = ppo_update(state, action, dist.log_prob(action), reward, next_state, done)

        state = next_state




while step < args.steps or not done:
    if done:
        seed = hash(seed) % (2 ** 31 - 1)
        env.reset(seed)
        done = False
    action = model(observation)

    sensor = model.encoder(observation)

    _, _, terminated, truncated, _ = env.step(action_space.sample())
    done = terminated or truncated
    step += 1
    bar.update(1)

