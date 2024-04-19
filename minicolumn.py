import torch
import torch.nn as nn
import torch.optim as optim

class MiniColumn(nn.Module):
    def __init__(self, in_size, latent_size, loss_function=nn.MSELoss):
        super(MiniColumn, self).__init__()
        self.in_size = in_size
        self.latent_size = latent_size
        self.loss_function = loss_function
        self.training_residual = None

        self.encoder = nn.Linear(self.in_size, self.latent_size)
        self.decoder = nn.Linear(self.latent_size, self.in_size)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)


    # def step(self, x):
    #     self.previous_input = x
    #     self.output = self.encoder(x)
    #     self.training_residual = self.decoder(self.output)
    #     return self.output, self.training_residual

    def train(self, next_input):
        self.optimizer.zero_grad()
        loss = self.loss_function(self.training_residual, next_input)
        loss.backward()
        self.optimizer.step()
        return loss

    def forward(self, x):
        output = self.encoder(x)
        self.training_residual = self.decoder(output)
        return output, self.training_residual



