import torch
import torch.nn as nn
import torch.optim as optim

class MiniColumn(nn.Module):
    def __init__(self, in_size, latent_size, loss_function=nn.MSELoss):
        super(MiniColumn, self).__init__()
        self.in_size = in_size
        self.latent_size = latent_size
        self.loss_function = loss_function()
        self.training_residual = None
        self.last_input = None

        self.encoder = nn.Linear(self.in_size, self.latent_size)
        self.encoder_activation = nn.ReLU()

        self.decoder = nn.Linear(self.latent_size, self.in_size)
        self.decoder_activation = nn.ReLU()

        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)


    # def step(self, x):
    #     self.previous_input = x
    #     self.output = self.encoder(x)
    #     self.training_residual = self.decoder(self.output)
    #     return self.output, self.training_residual

    # def train(self, next_input):
    #     self.optimizer.zero_grad()
    #     loss = self.loss_function(self.training_residual, next_input)
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss

    def initial_step(self, x):
        self.last_input = x
        output = self.encoder(x)
        self.training_residual = self.decoder(output)
        return output.detach(), self.training_residual

    def forward(self, x):
        # Train first
        self.optimizer.zero_grad()
        loss = self.loss_function(self.training_residual, x)
        loss.backward()
        self.optimizer.step()

        # Then step
        self.last_input = x
        output = self.encoder_activation(self.encoder(x))

        self.training_residual = self.decoder_activation(self.decoder(output))

        return output.detach(), self.training_residual



