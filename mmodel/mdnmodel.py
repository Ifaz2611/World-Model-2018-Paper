import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MModel(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=256, K=5, action_dim=0):
        super(MModel, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.K = K
        self.action_dim = action_dim

        self.lstm = nn.LSTM(input_size=self.latent_dim + self.action_dim, hidden_size=self.hidden_dim, batch_first=True)

        self.fc_mu = nn.Linear(in_features=self.hidden_dim, out_features=self.latent_dim * self.K)
        self.fc_pi = nn.Linear(in_features=self.hidden_dim, out_features=self.latent_dim * self.K)
        self.fc_logvar = nn.Linear(in_features=self.hidden_dim, out_features=self.latent_dim * self.K)


    def forward(self, z, a=None, hidden=None):
        X = z if a is None else torch.cat([z, a], dim=-1)

        output, hidden = self.lstm(X, hidden)
 
        batch, seq_len, _ = output.shape
        mixture_shape = (batch, seq_len, self.latent_dim, self.K)

        # pi is returned as logits; softmax over K happens in the loss/sampler
        # so the temperature can be applied there.
        pi_logits = self.fc_pi(output).view(mixture_shape)
        mu = self.fc_mu(output).view(mixture_shape)
        logvar = self.fc_logvar(output).view(mixture_shape)

        return pi_logits, mu, logvar, hidden

    def sample(self, pi_logits, mu, logvar, temperature=1.0):
        # Temperature tau flattens the mixture weights and widens each Gaussian,
        # which is how the paper controls how uncertain the dream rollouts are.
        k = torch.distributions.Categorical(logits=pi_logits / temperature).sample()
        k = k.unsqueeze(-1)

        mu_k = mu.gather(-1, k).squeeze(-1)
        logvar_k = logvar.gather(-1, k).squeeze(-1)
        std_k = torch.exp(0.5 * logvar_k) * math.sqrt(temperature)

        return mu_k + std_k * torch.randn_like(std_k)


def mdn_loss(pi_logits, mu, logvar, target):
    # target shape: (batch, seq_len, latent_dim) -- the next latent z_{t+1}.
    target = target.unsqueeze(-1)
    log_pi = F.log_softmax(pi_logits, dim=-1)

    log_prob = -0.5 * (math.log(2 * math.pi) + logvar + (target - mu) ** 2 / torch.exp(logvar))

    log_prob = torch.logsumexp(log_pi + log_prob, dim=-1)

    return -log_prob.sum(dim=-1).mean()
