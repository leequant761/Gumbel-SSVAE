# Main Reference : https://github.com/daandouwe/concrete-vae
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical

from pyro.distributions import Normal

from modules import Encoder, Decoder, Classifier

class Model(nn.Module):
    n_class = 10
    data_dim = 784
    def __init__(self, temp, latent_dim):
        super(Model, self).__init__()
        if type(temp) != torch.Tensor:
            temp = torch.tensor(temp)
        self.__temp = temp
        self.latent_dim = latent_dim
        self.encoder = Encoder(input=self.data_dim+self.n_class, latent_dim=latent_dim)
        self.decoder = Decoder(input=latent_dim+self.n_class)
        self.classifier = Classifier()
        self.prior_z = Normal(torch.zeros(latent_dim).cuda(), torch.ones(latent_dim).cuda()).to_event(1)
        self.prior_y = dist.RelaxedOneHotCategorical(temp, probs=torch.ones(self.n_class).cuda())
        self.initialize()

    @property
    def temp(self):
        return self.__temp

    @temp.setter
    def temp(self, value):
        self.__temp = value
        self.prior_y = dist.RelaxedOneHotCategorical(value, probs=torch.ones(self.n_class).cuda())

    def initialize(self):
        for param in self.parameters():
            if len(param.shape) > 2:
                nn.init.xavier_uniform_(param)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x, y=None):
        log_alpha = self.classifier(x)
        y_q, y_q_dist = self.sample_y(log_alpha, self.temp, y)
        z_param = self.encoder(torch.cat([x, y_q], axis=-1))
        z_q, z_q_dist = self.sample_z(z_param)
        x_recon = self.decode(torch.cat([z_q, y_q], axis=-1))
        return x_recon, z_q, y_q, z_q_dist, y_q_dist

    def approximate_loss(self, x, x_recon, z_q, z_q_dist, y_q_dist, is_observed, eps=1e-3):
        """ KL-divergence follows Eric Jang's trick
        """
        bce = F.binary_cross_entropy(x_recon, x.view(-1, self.data_dim), reduction='sum')
        n_batch = x.shape[0]
        prior_z = self.prior_z.expand(torch.Size([n_batch]))

        probs = y_q_dist.probs # alpha_i / alpha_sum
        if not is_observed:
            kl_y = torch.sum(probs * (self.n_class * (probs + eps)).log(), dim=-1).sum()
        else:
            kl_y = 0.
        # TODO January 24, 2021: kl_z can be derived analytically
        kl_z = (z_q_dist.log_prob(z_q) - prior_z.log_prob(z_q)).sum()
        kl = kl_y + kl_z
        return bce, kl

    def loss(self, x, x_recon, z_q, y_q, z_q_dist, y_q_dist, is_observed):
        """ Monte-Carlo estimate KL-divergence
        """
        bce = F.binary_cross_entropy(x_recon, x.view(-1, self.data_dim), reduction='sum')
        n_batch = x.shape[0]
        prior_y = self.prior_y.expand(torch.Size([n_batch]))
        prior_z = self.prior_z.expand(torch.Size([n_batch]))

        if not is_observed:
            kl_y = (y_q_dist.log_prob(y_q) - prior_y.log_prob(y_q)).sum()
        else:
            kl_y = 0.
        # TODO January 24, 2021: kl_z can be derived analytically
        kl_z = (z_q_dist.log_prob(z_q) - prior_z.log_prob(z_q)).sum()
        kl = kl_y + kl_z
        return bce, kl

    def sample_y(self, log_alpha, temp, y):
        v_dist = dist.RelaxedOneHotCategorical(temp, logits=log_alpha)
        concrete = v_dist.rsample() if y is None else y
        return concrete, v_dist

    def sample_z(self, v_param):
        v_mean, v_sd = v_param
        v_dist = Normal(v_mean, v_sd).to_event(1)
        z_q = v_dist.rsample()
        return z_q, v_dist