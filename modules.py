import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, input=50+10, hidden=200, output=784):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, output)
        self.act_fn = nn.Tanh()

    def forward(self, z):
        h = self.fc1(z)
        h = self.act_fn(h)
        return self.fc2(h)

class Encoder(nn.Module):
    def __init__(self, input=784, hidden=200, latent_dim=50):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc21 = nn.Linear(hidden, latent_dim)
        self.fc22 = nn.Linear(hidden, latent_dim)
        self.act_fn = nn.Tanh()
        self.latent_dim = latent_dim

    def forward(self, x):
        h = self.fc1(x)
        h = self.act_fn(h)
        mean, sd = self.fc21(h), self.fc22(h).exp()
        return mean, sd

class Classifier(nn.Module):
    def __init__(self, input=784, hidden=200, output=10):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, output)
        self.act_fn = nn.Tanh()

    def forward(self, z):
        h = self.fc1(z)
        h = self.act_fn(h)
        return self.fc2(h)