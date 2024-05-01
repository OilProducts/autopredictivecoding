import argparse
import os

import torch
from torchvision.utils import save_image

import gymnasium as gym


def main(args):
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)  # Create output directory if it does not exist

    env = gym.make(f"ALE/{args.env}-v5", render_mode='rgb_array_list')  # Create the game
    # environment

    steps = 0
    state, _ = env.reset()
    while steps < args.steps:
        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)
        state_tensor = torch.from_numpy(state).float() / 255.0
        state_tensor = state_tensor.permute(2, 0, 1)
        save_image(state_tensor, f'{args.outdir}/{args.env}-{steps}.png')

        steps += 1
        if terminated or truncated:
            state, _ = env.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='./data/')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--env', type=str, default='Pong')
    args = parser.parse_args()

    main(args)
