import argparse

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

