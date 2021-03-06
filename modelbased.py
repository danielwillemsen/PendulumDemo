import time
import torch
import copy
import numpy as np
from utils import ReplayBuffer
from utils import Critic
from utils import Actor
from utils import Model


class ModelAgent:
    def __init__(self, obs_dim, action_dim, *args, **kwargs):
        # Initialize arguments
        hidden_dims_actor = tuple(kwargs.get("hidden_dims_actor",
                                             (256, 256)))
        hidden_dims_critic = tuple(kwargs.get("hidden_dims_critic",
                                              (256, 256)))
        hidden_dims_model = tuple(kwargs.get("hidden_dims_model",
                                             (256, 256)))

        self.gamma = 0.99
        self.tau = 0.005
        self.delay = 2
        lr_actor = 0.001
        lr_critic = 0.001
        lr_model = 0.0001
        self.step_random = 500 # How many random actions to take before using actor for action selection
        self.update_every_n_steps = 51 # How often to update model, actor and critics
        self.update_steps = 200 # How many gradient updates to perform, per model,  when updating
        self.time = time.time()

        # Initialize actor
        self.actor = Actor(obs_dim, hidden_dims_actor,  action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(),
                                                lr=lr_actor)
        for par in self.actor_target.parameters():
            par.requires_grad = False

        # Initialize 2 critics
        self.critics = []
        self.critics_target = []
        self.optimizer_critics = []
        for k in range(2):
            critic = Critic(obs_dim + action_dim, hidden_dims_critic)
            self.critics.append(critic)
            self.critics_target.append(copy.deepcopy(critic))
            self.optimizer_critics.append(torch.optim.Adam(critic.parameters(),
                                                           lr=lr_critic))

            for par in self.critics_target[k].parameters():
                par.requires_grad = False

        # Initialize models
        self.models = []
        self.optimizer_models = []
        for k in range(25):
            model = Model(obs_dim + action_dim, hidden_dims_model, obs_dim)
            self.models.append(model)
            self.optimizer_models.append(torch.optim.Adam(model.parameters(),
                                                          lr=lr_model))

        # Setup Replay Buffer
        self.buffer = ReplayBuffer()
        self.o_old = None
        self.a_old = None

        self.step_i = 0

    def reset(self):
        self.o_old = None
        self.a_old = None

    def loss_critic(self, val, target):
        diffs = target - val
        return torch.mean(diffs**2)

    def step(self, o, r, eval=False, done=False):
        o = torch.Tensor(o)
        r = torch.Tensor(np.array(float(r)))
        done = torch.Tensor(np.array(float(done)))
        if not eval:
            # Add to replay buffer
            if self.o_old is not None:
                self.buffer.add((self.o_old, self.a_old, r, o, done))

            if self.step_i % self.update_every_n_steps == 0 and self.buffer.len() > self.buffer.n_samples:
                print("Performing update steps...")
                for step in range(self.update_steps):
                    # Train Model
                    self.update_models()

                    # Train actor and critics.
                    # Using one minibatch of samples from every model.
                    for model in self.models:
                        b = self.sample_from_model(model)

                        # Update Critic
                        self.update_critics(b)

                        # Update Actor
                        if self.step_i % self.delay == 0:
                            self.update_actor(b)

                            # Update Target Networks
                            self.update_target_networks()

            # Select Action
            with torch.no_grad():
                action = self.actor(o.unsqueeze(0)).squeeze()
                action_noisy = action + torch.randn(action.size())*0.3
                action = torch.clamp(action_noisy, 0., 1.0)
            if self.step_i < self.step_random:
                action = torch.rand(action_noisy.shape)
            self.o_old = o
            if action.size() == torch.Size([]):
                self.a_old = action.unsqueeze(0)
            else:
                self.a_old = action
            self.step_i += 1
        else:
            action = self.actor(o.unsqueeze(0)).squeeze()
            action = torch.clamp(action, 0., 1.0)

        return action.detach().numpy()

    def update_target_networks(self):
        with torch.no_grad():
            for par, par_target in zip(self.actor.parameters(), self.actor_target.parameters()):
                par_target.data.copy_(
                    (1 - self.tau) * par_target + self.tau * par.data)
            for k in range(2):
                for par, par_target in zip(self.critics[k].parameters(), self.critics_target[k].parameters()):
                    par_target.data.copy_(
                        (1 - self.tau) * par_target + self.tau * par.data)

    def update_actor(self, b):
        for par in self.critics[0].parameters():
            par.requires_grad = False
        self.optimizer_actor.zero_grad()
        loss_actor = - \
            torch.mean(self.critics[0](b["o"], self.actor(b["o"])))
        loss_actor.backward()
        self.optimizer_actor.step()
        for par in self.critics[0].parameters():
            par.requires_grad = True

    def update_critics(self, b):
        with torch.no_grad():
            a_target = self.actor_target(b["o_next"])
            a_target = torch.clamp(
                a_target + torch.clamp(torch.randn(a_target.size()) * 0.1, -0.5, 0.5), 0., 1.)
            y = b["r"].unsqueeze(-1) + (1 - b["done"]) * self.gamma * torch.min(
                *[critic_target(b["o_next"], a_target) for critic_target in self.critics_target])
        for optimizer, critic in zip(self.optimizer_critics, self.critics):
            loss = self.loss_critic(critic(b["o"], b["a"]), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def sample_from_model(self, model):
        # Sample Minibatch
        b = self.buffer.sample_tensors(n=128)
        with torch.no_grad():
            action = self.actor(b["o"])
            action_noisy = action + torch.randn(action.size()) * 0.3
            b["a"] = torch.clamp(action_noisy, 0., 1.0)
        new_o, r = model.sample(b["o"], b["a"])
        b["o_next"] = new_o
        b["r"] = r.squeeze()
        return b

    def update_models(self):
        samples = self.buffer.sample_tensors()
        for optim, model in zip(self.optimizer_models, self.models):
            self.model_step(model, optim, samples)

    def model_step(self, model, optim, samples):
        # Do one gradient update for a model.
        o_next_pred, r_pred = model(samples["o"], samples["a"])
        sigma = o_next_pred[1]
        sigma_2 = r_pred[1]
        mu = o_next_pred[0]
        target = samples["o_next"]
        loss1 = torch.mean((mu - target) / sigma ** 2 * (mu - target))
        loss2 = torch.mean(torch.log(torch.prod(sigma ** 2, 1) * torch.prod(sigma_2 ** 2, 1)))
        mu = r_pred[0]
        target = samples["r"].unsqueeze(1)
        loss3 = torch.mean((mu - target) / sigma_2 ** 2 * (mu - target))
        loss = loss1 + loss2 + loss3
        optim.zero_grad()
        loss.backward()
        optim.step()
