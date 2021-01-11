import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import collections
import random
import replay

class DQN:

    def __init__(self, network, action_space, replay_length=10000, gamma=1.0, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        # Parameters
        self.action_space = action_space

        # Hyperparameters
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min 

        # Objects
        self.replay_memory = replay.PrioritizedMemory(replay_length)
        self.network = network 

    def act(self, state):
        if (random.random() < self.epsilon):
            return random.randint(0, self.action_space-1)

        action = torch.argmax(self.network.forward(state)).item()
        return action

    def remember(self, error, state, action, reward, next_state, done):
        self.replay_memory.insert(error, (state, action, reward, next_state, done))

    def compute_q(self, state, reward, next_state, done):
        return reward if done else reward + self.gamma * torch.max(self.network.forward(next_state)).item()

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        batch = random.sample(self.replay_memory, min(len(self.replay_memory), batch_size))
        for state, action, reward, next_state, done in batch:
            y = self.network.forward(state)
            y[action] = self.compute_q(state, reward, next_state, done)            
            x_batch.append(state)
            y_batch.append(y)

        self.network.train(torch.Tensor(x_batch), torch.stack(y_batch))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run(self, env, batch_size, solved_count=np.inf, max_episodes=10000):
        rewards = collections.deque(maxlen=100)
        for _ in range(max_episodes):
            done =  False
            state = env.reset()
            total_rewards = 0
            for step in range(201):
                action = self.act(state)
                next_state, reward, done, info = env.step(action)
                reward = reward if not done else -10
                total_rewards += reward
                if not self.replay_memory.is_full():
                    pred = 0
                else:
                    pred = self.compute_q(state, reward, next_state, done)

                self.remember(np.abs(), state, action, reward, next_state, done)
                state = next_state
                if done:
                    if (_ % 25) == 0:
                        print("Episode {}, Score={}".format(_, total_rewards))
                    break
                
                step += 1

                if len(self.replay_memory) >= batch_size:
                    self.replay(batch_size)

            rewards.append(total_rewards)
            if sum(rewards) / len(rewards) >= solved_count:
                print("Solved, episode {}".format(_))

