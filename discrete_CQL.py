# discrete_CQL.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state):
        return self.net(state)


class discrete_CQL:
    def __init__(self, is_atari, num_actions, state_dim, device, alpha, discount,
                 optimizer, optimizer_parameters, polyak_target_update, target_update_freq,
                 tau, initial_eps, end_eps, eps_decay_period, eval_eps):

        self.device = device
        self.action_dim = num_actions
        self.discount = discount
        self.alpha = alpha
        self.eps = eval_eps
        self.total_it = 0
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.polyak = polyak_target_update

        self.Q = QNetwork(state_dim, num_actions).to(device)
        self.Q_target = QNetwork(state_dim, num_actions).to(device)
        self.Q_target.load_state_dict(self.Q.state_dict())

        self.optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

    def select_action(self, state, eval=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            q_values = self.Q(state)
            return q_values.argmax().item()

    def train(self, replay_buffer):
        self.total_it += 1

        # Sample a batch from replay buffer
        state, action, next_state, reward, done = replay_buffer.sample()

        with torch.no_grad():
            next_q = self.Q_target(next_state)
            max_next_q = torch.max(next_q, dim=1)[0]
            target_q = reward + (1 - done) * self.discount * max_next_q

        current_q = self.Q(state).gather(1, action.unsqueeze(1)).squeeze(1)
        bellman_loss = F.mse_loss(current_q, target_q)

        # CQL regularization: penalize unseen Q-values
        all_q = self.Q(state)
        logsumexp_q = torch.logsumexp(all_q, dim=1)
        dataset_q = current_q
        cql_loss = (logsumexp_q - dataset_q).mean()

        total_loss = bellman_loss + self.alpha * cql_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Soft update
        if self.polyak:
            for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        elif self.total_it % self.target_update_freq == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

        return {
            "total_loss": total_loss.item(),
            "q_loss": bellman_loss.item(),
            "regularizer": cql_loss.item(),
            "q_values_mean": all_q.mean().item(),
            "target_q_mean": target_q.mean().item(),
        }

    def save(self, filename):
        torch.save(self.Q.state_dict(), filename)

    def load(self, filename):
        self.Q.load_state_dict(torch.load(filename))
        self.Q_target.load_state_dict(self.Q.state_dict())
