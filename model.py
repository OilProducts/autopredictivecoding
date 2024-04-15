from typing import Union, Callable, Tuple, Any, Optional, Dict

import torch
import torch.nn as nn
from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle


# Each logical portion of the model needs to be an autoencoder of some sort. The latent space of the autoencoder
# is part of its own input.  The decoder should be trained in parallel.  Two experiments come to mind, one is having the
# decoder recreate the current observation, the other is having it recreate the next time steps observation.  It should
# not need to recreate the latent space input.  The previous applies to the sensor
#
# The model should have at least the following logical portions:
#   - The sensor
#   - The mini columns
#   - The motor units


# -----  Input Vector
#  ---
#   -    Output
#  ---
# -----  Training residual, unused except for loss calculation


# The sensor should take the observation as input, and output the latent space of the autoencoder. With loss calculated
# against the reconstruction of the observation.

# The mini columns input vector should be other units latent space vectors (Either sensors or other mini columns). The
# training residual should be the *NEXT TIME STEP INPUT VECTOR* (This is important, as it will allow the mini columns to
# predict the next time step).

# The motor units should take the latent space of the mini columns as input, and output the action space.  The training
# residual is unclear.  Its possible that in gym environments we don't need to actually consider motor units to be
# significant neuron groups.  Instead these might be considered some structure like the thalamus.  In this case the
# training residual could be related to the next time steps input vector, or some kind of external reward signal.
#
# In the context of a gym environment we need
# to take the observation, and the latent space of the autoencoder as input.

# Sensors may not need to take their output as input, but mini columns should

# Motor units should take the latent space as input, and output the action space


class NeuronUnit(torch.nn.Module):
    def __init__(self, loss_function=nn.MSELoss):
        super(NeuronUnit, self).__init__()

        self.loss_function = loss_function
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # output: 16 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # output: 32 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # output: 64 x 8 x 8
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # output: 128 x 4 x 4
            nn.ReLU(),
            nn.Flatten(),  # output: 2048
            nn.Linear(2048, 256),  # output: 256 (encoded space)
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 2048),  # output: 2048
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),  # output: 128 x 4 x 4
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            # output: 64 x 8 x 8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            # output: 32 x 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            # output: 16 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            # output: 3 x 64 x 64
            nn.Sigmoid()  # Using Sigmoid to scale the output between 0 and 1
        )

        self.training_residual = None

        # self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        output = self.encoder(x)
        training_residual = self.decoder(output)
        return output, training_residual


class MiniColumn(torch.nn.Module):
    def __init__(self, input, output):
        super(MiniColumn, self).__init__()
        self.fc = torch.nn.Linear(input, output)

    def forward(self, x):
        return self.fc(x)



