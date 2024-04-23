import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Sensor(nn.Module):
    def __init__(self, loss_function=nn.MSELoss):
        super(Sensor, self).__init__()
        self.loss_function = loss_function()
        self.training_residual = None

        # Encoder layers
        self.enc_conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc_conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(17920, 256)  # Assuming the input to this layer is 17920

        # Decoder layers
        self.fc2 = nn.Linear(256, 17920)
        self.unflatten = nn.Unflatten(1, (128, 14, 10))
        self.dec_convtrans1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec_convtrans2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.dec_convtrans3 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.dec_convtrans4 = nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1)
        self.sigmoid = nn.Sigmoid()


        # Encoder, input is 210 x 160 x 3
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 16, 3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, 3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, 3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(17920, 256),  # output: 256 (encoded space)
        #     nn.ReLU()
        # )
        #
        # # Decoder
        # self.decoder = nn.Sequential(
        #     nn.Linear(256, 17920),  # output: 2048
        #     nn.ReLU(),
        #     nn.Unflatten(1, (128, 14, 10)),  # output: 128 x 4 x 4
        #     nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
        #     # output: 64 x 8 x 8
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
        #     # output: 32 x 16 x 16
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        #     # output: 16 x 32 x 32
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
        #     # output: 3 x 64 x 64
        #     nn.Sigmoid()  # Using Sigmoid to scale the output between 0 and 1
        # )

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        # self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        """ For a sensor being trained against the current time step
        we run the forward pass and then train.  For a sensor being trained
        against the next time step we run the forward pass once, then for
        every other time step we run the train first, with the curren time
        steps input."""
        # Encoder forward pass
        y = F.relu(self.enc_conv1(x))
        y = F.relu(self.enc_conv2(y))
        y = F.relu(self.enc_conv3(y))
        y = F.relu(self.enc_conv4(y))
        y = self.flatten(y)
        y = self.fc1(y)
        output = F.relu(y)

        # Decoder forward pass
        z = F.relu(self.fc2(output))
        z = self.unflatten(z)
        z = F.relu(self.dec_convtrans1(z))
        z = F.relu(self.dec_convtrans2(z))
        z = F.relu(self.dec_convtrans3(z))
        self.training_residual = self.sigmoid(self.dec_convtrans4(z))

        self.optimizer.zero_grad()
        loss = self.loss_function(self.training_residual, x)
        loss.backward()
        self.optimizer.step()
        return output, self.training_residual

    # def train(self, input):
    #     self.optimizer.zero_grad()
    #     loss = self.loss_function(self.training_residual, input)
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss

    # To test whether the sensor should also predict the next time step
    # def train(self, next_input):
    #     self.optimizer.zero_grad()
    #     loss = self.loss_function(self.training_residual, next_input)
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss