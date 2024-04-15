import torch

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
    def __init__(self, input_size, output_size):
        super(NeuronUnit, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)