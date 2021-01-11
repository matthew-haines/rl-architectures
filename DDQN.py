import gym
import torch
import numpy as np
import random
from networkscpu import BasicDNN
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import gc

writer = SummaryWriter()

class Agent:

    def __init__(self, state_size, action_size, hidden_size=256, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.99, gamma=0.9, memory_length=10000, batch_size=32, update_freq=500):
        self.state_size = state_size
        self.action_size = action_size

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.gamma = gamma
        self.update_freq = update_freq

        self.batch_size = batch_size

        self.memory_length = memory_length
        self.memory = deque(maxlen=memory_length)

        self.network = BasicDNN(self.state_size, hidden_size, self.action_size)
        self.target_network = BasicDNN(self.state_size, hidden_size, self.action_size)

        self.train_count = 0

    def _update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_action(self, state):
        return random.randint(0, self.action_size-1) if random.random() <= self.epsilon else np.argmax(self.network.forward(state))

    def _update_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        samples = random.sample(self.memory, self.batch_size)
        x = []
        y = []
        for state, action, reward, next_state, done in samples:
            x.append(state)
            y_temp = self.network.forward(state)
            y_temp[action] = reward if done else reward + self.gamma * \
                self.target_network.forward(next_state)[np.argmax(self.network.forward(next_state))]

            y.append(y_temp)

        x = np.stack(x)
        y = np.stack(y)
        loss, _ = self.network.train(x, y)
        
        writer.add_scalar('Loss', loss, self.train_count)
        self.train_count += 1

        self._update_epsilon()

    def run(self, env, episodes, episode_len):
        step_count = 0
        all_reward = []
        for e in range(episodes):
            state = env.reset()
            total_reward = 0
            for step in range(episode_len):
                env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                total_reward += reward
                
                if step_count % self.update_freq:
                    self._update_network()

                step_count += 1

                if len(self.memory) >= self.batch_size:
                    self.replay()

                if done:
                    break

                state = next_state
            
            writer.add_scalar('Reward', total_reward, e)
            avg = sum(all_reward[-20:]) / 20 
            if e % 10 == 0:
                print('Episode: {}, Reward: {}'.format(e, avg))

            all_reward.append(total_reward)

            if avg > 180:
                torch.save(self.network.state_dict(), f'saved_models/{e}.dat')
            
            gc.collect()


if __name__ == '__main__':
    torch.random.manual_seed(2)
    np.random.seed(2)

    env = gym.make('CartPole-v0')
    state_space = 4
    action_space = 2
    agent = Agent(state_space, action_space)
    agent.run(env, 1000, 200)
