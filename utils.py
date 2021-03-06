import torch
import random

from torch import nn


class ReplayBuffer():
    def __init__(self):
        self.buffer = []
        self.n_samples = 128
        self.max_size = 1000000

    def len(self):
        return len(self.buffer)

    def add(self, sample):
        self.buffer.append(sample)
        if len(self.buffer) > self.max_size:
            del self.buffer[0]

    def sample(self):
        samples = random.choices(self.buffer, k=self.n_samples)
        data = [*zip(samples)]
        data_dict = {"o": data[0], "a": data[1], "r": data[2], "o_next": data[3], "done": data[3]}
        return data_dict

    def sample_tensors(self, n=128):
        samples = random.choices(self.buffer, k=n)
        data = [*zip(*samples)]
        data_dict = {"o": torch.stack(data[0]), "a": torch.stack(data[1]), "r": torch.stack(data[2]), "o_next": torch.stack(data[3]), "done": torch.stack(data[4])}
        return data_dict


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims_actor=(164, 164), hidden_dims_critic=(164,164)):
        super().__init__()
        self.actor = Actor(obs_dim, hidden_dims_actor, action_dim)
        self.critic = Critic(obs_dim + action_dim, hidden_dims_critic)


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        layers += [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims)-1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        layers += [nn.Linear(hidden_dims[-1], output_dim), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, observation):
        return self.net(observation)


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        layers += [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        layers += [nn.Linear(hidden_dims[-1], 1), nn.Identity()]
        self.net = nn.Sequential(*layers)

    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=-1)
        return self.net(x)


class Model(nn.Module):
    """Model.
    Contains a probabilistic world model. Outputs 2 lists: one containing mu, sigma of reward, second containing mu, sigma of observation
    """
    def __init__(self, input_dim, hidden_dims, obs_dim):
        """__init__.

        :param input_dim:
        :param hidden_dims:
        :param output_dim:
        """
        super().__init__()
        layers = []
        layers += [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        self.net = nn.Sequential(*layers)
        self.mu_output = nn.Linear(hidden_dims[-1], obs_dim)
        self.sigma_output = nn.Linear(hidden_dims[-1], obs_dim)
        self.mu_reward = nn.Linear(hidden_dims[-1], 1)
        self.sigma_reward = nn.Linear(hidden_dims[-1], 1)

    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=-1)
        x = self.net(x)
        return [self.mu_output(x), torch.exp(self.sigma_output(x))], [self.mu_reward(x)*5, torch.exp(self.sigma_reward(x))]

    def sample(self, observation, action):
        with torch.no_grad():
            new_o, r = self.forward(observation, action)
            new_o = torch.normal(new_o[0], new_o[1])
            r = torch.normal(r[0], r[1])
        return new_o, r
