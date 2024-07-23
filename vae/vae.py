import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    def __init__(self, latent_dims, input_height, input_width):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(16),
            nn.Conv2d(16, 24, 3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(24),
            nn.Conv2d(24, 32, 3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(32)
        )

        conv_output_size = self._get_conv_output(input_height, input_width)
        self.fc_mu = nn.Linear(in_features=conv_output_size, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=conv_output_size, out_features=latent_dims)

    def _get_conv_output(self, input_height, input_width):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_height, input_width)
            output = self.conv_layers(dummy_input)
        return int(np.prod(output.size()[1:]))  # [1:] to discard the batch dimension

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self, latent_dims, output_height, output_width):
        super(Decoder, self).__init__()
        self.output_height, self.output_width = output_height // 8, output_width // 8  # Adjust according to your conv/deconv layers

        self.fc = nn.Sequential(
            nn.Linear(latent_dims, 32 * self.output_height * self.output_width), nn.ReLU()
        )

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(32, 24, 3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(24),
            nn.ConvTranspose2d(24, 24, 3, stride=2, padding=1, output_padding=1), nn.ReLU(), nn.BatchNorm2d(24),
            nn.ConvTranspose2d(24, 16, 3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(), nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 8, 3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1, output_padding=1), nn.ReLU(), nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 1, 3, stride=1, padding=1), nn.Sigmoid()
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((output_height, output_width))


    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 32, self.output_height, self.output_width)
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)  # Adjust the size to match the target dimensions

        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, variational_beta=1.0, input_height=513, input_width=100):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dims, input_height, input_width)
        self.decoder = Decoder(latent_dims, input_height, input_width)
        self.variational_beta = variational_beta

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def loss(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.variational_beta * kldivergence
