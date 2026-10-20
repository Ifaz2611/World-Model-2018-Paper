import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.latent_dim = 32

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(in_features=1024, out_features=self.latent_dim)
        self.fc_logvar = nn.Linear(in_features=1024, out_features=self.latent_dim)
        self.fc_expand = nn.Linear(in_features=self.latent_dim, out_features=1024)

        self.decoder_reconstructor = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2),
            nn.Sigmoid()
        )

    def reparameterization(self, mu, logvar) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    
    def encode(self, X):
        # The M model only ever needs this half: frames in, latent distribution out,
        # with the decoder frozen and unused.
        features = self.feature_extractor(X)

        return self.fc_mu(features), self.fc_logvar(features)

    def forward(self, X):
        # X shape: (channels=3, width=64, height=64)
        mu, logvar = self.encode(X)
        sample = self.reparameterization(mu, logvar)

        X_recon = self.fc_expand(sample)
        X_recon = X_recon.view(X_recon.shape[0], X_recon.shape[1], 1, 1)

        X_recon = self.decoder_reconstructor(X_recon)

        return X_recon, X, mu, logvar

    def generate(self, z):
        ...


def elbo_loss(x_recon, x, mu, logvar):
    batch_size = x.shape[0]

    # Both terms are summed over their dimensions and averaged over the batch,
    # so the reconstruction and KL stay on a comparable scale.
    reconstruct_loss = F.mse_loss(x_recon, x, reduction='sum') / batch_size

    # KL(q(z|x) || N(0, I))
    kl_loss = 0.5 * torch.sum(mu**2 + torch.exp(logvar) - 1 - logvar) / batch_size

    return reconstruct_loss + kl_loss

