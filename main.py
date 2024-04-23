import argparse
import time
import torch
from torch.distributions import Categorical

import tqdm

import gymnasium as gym
from gymnasium.utils.save_video import save_video
# import crafter

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
#
# env = crafter.Env()
# env = crafter.Recorder(
#     env, args.outdir,
#     save_stats=True,
#     save_episode=False,
#     save_video=False,
# )
# action_space = env.action_space
#
# # model = model.Brain()
#
# done = True
# step = 0
# bar = tqdm.tqdm(total=args.steps, smoothing=0)
# state = env.reset()
#
# observation, _, terminated, truncated, _ = env.step(action_space.sample())

n_episodes = 500


def main():
    env = gym.make("ALE/Breakout-v5", render_mode='rgb_array_list')  # Create the game environment
    brain = model.Brain(210 * 160 * 3, 4)
    total_score = 0.0
    total_steps = 0

    for episode in range(n_episodes):
        steps = 0
        start_time = time.time()
        state, _ = env.reset()
        brain.reset()
        state = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0)

        # First round, so we don't end up storing the initial state along with 0 reward
        action = brain.initial_action(state)

        state, reward, terminated, truncated, _ = env.step(action)
        total_steps += 1
        steps += 1
        state = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0)

        done = False
        score = 0.0
        while not done:
            action = brain(state, reward, done)
            state, reward, terminated, truncated, _ = env.step(action)
            total_steps += 1
            steps += 1
            state = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0)
            done = terminated or truncated
            score += reward
        action = brain(state, reward, done)

        end_time = time.time()
        total_score += score
        print(
            f'Episode {episode} finished in {end_time - start_time:.2f} seconds and {steps} steps, '
            f'score: {score:.1f}, total steps: {total_steps}')

        if score > 20:
            save_video(env.render(),
                       "videos",
                       fps=30,
                       episode_trigger=lambda episode_id: True,
                       step_starting_index=0,
                       episode_index=episode, )

        save_video(env.render(),
                   "videos",
                   fps=30,
                   step_starting_index=0,
                   episode_index=episode, )

    env.close()
    # while step < args.steps or not done:
    #     if done:
    #         seed = hash(seed) % (2 ** 31 - 1)
    #         env.reset(seed)
    #         done = False
    #     action = model(observation)
    #
    #     sensor = model.encoder(observation)
    #
    #     _, _, terminated, truncated, _ = env.step(action_space.sample())
    #     done = terminated or truncated
    #     step += 1
    #     bar.update(1)


if __name__ == '__main__':
    main()
