from typing import Union, Callable, Tuple, Any, Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
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

import sensor, minicolumn, motor_unit, ppo, decision
class Brain(nn.Module):
    def __init__(self, in_sz, out_sz):
        super(Brain, self).__init__()
        self.sensor = sensor.SensorAtari()
        self.mini_column1 = minicolumn.MiniColumn(512, 192)
        self.mini_column2 = minicolumn.MiniColumn(704, 192)
        self.motor_unit = decision.PPOSplit(896, out_sz)
        self.latent_state = None

    def forward(self, x, reward, done):
        sensor_output, sensor_residual = self.sensor(x)
        sensor_output = sensor_output.squeeze()
        mini_column1_output, mini_column1_residual = self.mini_column1.initial_step(sensor_output)
        mini_column1_output = mini_column1_output.squeeze()

        mini_column2_output, mini_column2_residual = self.mini_column2.initial_step(
            torch.cat((sensor_output, mini_column1_output), 0))
        mini_column2_output = mini_column2_output.squeeze()

        self.latent_state = torch.cat((sensor_output, mini_column1_output, mini_column2_output), 0)
        motor_output = self.motor_unit.act_and_train(self.latent_state, reward, done)
        return motor_output

    def initial_action(self, x):
        sensor_output, sensor_residual = self.sensor(x)
        sensor_output = sensor_output.squeeze()
        mini_column1_output, mini_column1_residual = self.mini_column1.initial_step(sensor_output)
        mini_column1_output = mini_column1_output.squeeze()

        mini_column2_output, mini_column2_residual = self.mini_column2.initial_step(
            torch.cat((sensor_output, mini_column1_output), 0))
        mini_column2_output = mini_column2_output.squeeze()

        self.latent_state = torch.cat((sensor_output, mini_column1_output, mini_column2_output), 0)
        motor_output = self.motor_unit.act(self.latent_state)
        return motor_output

    def reset(self):
        # self.sensor.reset()
        # self.mini_column.reset()
        self.motor_unit.reset()
        self.latent_state = None