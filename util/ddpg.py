"""Implementation of Deep Deterministic Policy Gradient (DDPG).

Adapted from https://github.com/lily-x/mirror/blob/main/ddpg.py
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, input_size, output_size):
        hidden_size1 = 16
        hidden_size2 = 32
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size2)
        self.linear4 = nn.Linear(hidden_size2, output_size)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        return x


class Actor(nn.Module):
    def __init__(self, input_size, output_size, learning_rate=3e-4):
        hidden_size1 = 16
        hidden_size2 = 32
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size2)
        self.linear4 = nn.Linear(hidden_size2, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        x = self.linear1(state)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = self.softmax(x)
        return x


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([experience[0] for experience in batch])
        a_batch = np.array([experience[1] for experience in batch])
        r_batch = np.array([experience[2] for experience in batch])
        ss_batch = np.array([experience[3] for experience in batch])
        done_batch = np.array([experience[4] for experience in batch])

        return s_batch, a_batch, r_batch, ss_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class NormalizedEnv:
    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low) / 2.0
        act_b = (self.action_space.high + self.action_space.low) / 2.0
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2.0 / (self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low) / 2.0
        return act_k_inv * (action - act_b)


class DDPG:
    def __init__(
        self,
        actions_dim,
        states_dim,
        actor_learning_rate=1e-4,
        critic_learning_rate=1e-3,
        gamma=0.99,
        tau=1e-2,
        memory_max_size=50000,
        batch_size=128,
    ):
        self.actions_dim = actions_dim
        self.states_dim = states_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.actor = Actor(self.states_dim, self.actions_dim)
        self.critic = Critic(self.states_dim + self.actions_dim, 1)

        self.actor_target = Actor(self.states_dim, self.actions_dim)
        self.critic_target = Critic(self.states_dim + self.actions_dim, 1)

        for target_param, param in zip(
            self.actor_target.parameters(),
            self.actor.parameters(),
        ):
            target_param.data.copy_(param.data)

        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic.parameters(),
        ):
            target_param.data.copy_(param.data)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_learning_rate,
        )

        self.memory = ReplayBuffer(memory_max_size)
        self.loss = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def select_action(self, state):
        if not torch.is_tensor(state):
            state = torch.from_numpy(state).float().unsqueeze(0)
        else:
            state = state.float().unsqueeze(0)
        action = self.actor.forward(state)
        action = action.detach().numpy()[0]
        return action

    def update(self):
        if len(self.memory) <= self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size,
        )
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # update critic by minimizing loss
        Q_vals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Q_prime = rewards + self.gamma * next_Q

        critic_loss = self.loss(Q_vals, Q_prime)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor policy using sampled policy gradient
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update target networks (slowly using soft updates)
        for target_param, param in zip(
            self.actor_target.parameters(),
            self.actor.parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data,
            )

        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic.parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data,
            )
