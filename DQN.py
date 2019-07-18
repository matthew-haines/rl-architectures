import torch
import torch.nn as nn 
import torch.nn.functional as F 
import gym
import numpy as np
import matplotlib.pyplot as plt
import collections
import random

class DQN:

    def __init__(self, network, action_space, replay_length=100000, gamma=1.0, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        # Parameters
        self.action_space = action_space

        # Hyperparameters
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min 

        # Objects
        self.replay_memory = collections.deque(maxlen=replay_length)
        self.network = network 

    def act(self, state):
        if (random.random() < self.epsilon):
            return random.randint(0, self.action_space-1)

        return self.network.forward(state)

    def remember(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        batch = random.sample(self.replay_memory, min(len(self.replay_memory), batch_size))
        for state, action, reward, next_state, done in batch:
            y = self.network.forward(state)
            y[0][action] = reward if done else reward + self.gamma * np.max(self.network.forward(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y[0])

        self.model.train(torch.Tensor(x_batch), torch.Tensor(y_batch))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run(self, env, max_episodes=10000):
        
        for _ in range(max_episodes):
            