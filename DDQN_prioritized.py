import gym
import torch
import numpy as np
import random
from networks import BasicDNNPyTorch as BasicDNN
from replay import PrioritizedMemory
from torch.utils.tensorboard import SummaryWriter
import gc

writer = SummaryWriter()


class Agent:

    def __init__(self, state_size, action_size, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01, gamma=0.99, memory_len=10000, batch_size=32, update_freq=1000):

        self.state_size = state_size
        self.action_size = action_size

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.gamma = gamma
        self.update_freq = update_freq

        self.batch_size = batch_size

        self.memory_len = memory_len
        self.memory = PrioritizedMemory(maxlen=memory_len)

        self.network = BasicDNN(self.state_size, self.action_size).cuda()
        self.target_network = BasicDNN(self.state_size, self.action_size).cuda()

        self.train_count = 0
        self.episodes_complete = 0

    def _update_networks(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def _epsilon_update(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _get_action(self, state):
        return random.randint(0, self.action_size-1) if random.random() <= self.epsilon else torch.argmax(self.network.forward(state))

    def _get_target(self, reward, next_state, done):
        target = reward if done else reward + self.gamma * self.target_network.forward(next_state)[torch.argmax(self.network.forward(next_state)).item()]
        return target

    def remember(self, state, action, reward, next_state, done):
        if self.network.is_training:
            error = torch.abs(self.network.forward(state)[action]-self._get_target(reward, next_state, done)).item()
        else:
            error = abs(reward)

        self.memory.insert(error, (state, action, reward, next_state, done))

    def replay(self):
        samples = [self.memory.retrieve() for i in range(self.batch_size)] # List of data, index
        x = []
        y = []
        update = []
        for sample in samples:
            state, action, reward, next_state, done = sample[0]
            index = sample[1]
            x.append(state)
            y_sample = self.network.forward(state)
            pred = y_sample[action]
            y_sample[action] = self._get_target(reward, next_state, done)
            y.append(y_sample)
            error = torch.abs(pred-y_sample[action]).item()
            update.append((index, error))

        x = torch.stack(x)
        y = torch.stack(y)
        loss = self.network.train(x, y)

        writer.add_scalar('Loss', loss, self.train_count)
        self.train_count += 1

        if self.train_count % self.update_freq == 0:
            self._update_networks()

        for index, error in update:
            self.memory.update(index, error)

    def run(self, env, episodes, max_steps):
        for e in range(episodes):
            total_reward = 0
            state = torch.Tensor(env.reset())
            for step in range(max_steps):
                action = self._get_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = torch.Tensor(next_state)
                self.remember(state, action, reward, next_state, done)

                total_reward += reward

                if self.memory.is_full():
                    self.replay()

                if done:
                    break

                state = next_state


            writer.add_scalar('Reward', total_reward, self.episodes_complete)
            self.episodes_complete += 1

            gc.collect()

if __name__ == '__main__':
    torch.random.manual_seed(2)
    np.random.seed(2)

    env = gym.make('CartPole-v0')
    agent = Agent(4, 2)
    agent.run(env, 1000, 200)