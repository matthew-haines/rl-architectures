import torch
import collections
import random 
import numpy as np 
from copy import deepcopy

# target_policy <- tau * base_policy + (1 - tau) * target_policy

class DQN:

    def __init__(self, network, action_space, replay_length=10000, gamma=1.0, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.005, tau=0.001):
        # Parameters
        self.action_space = action_space

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau 

        # Objects
        self.replay_memory = collections.deque(maxlen=replay_length)
        self.network = network
        self.target_network = deepcopy(self.network)

    def act(self, state):
        if (random.random() < self.epsilon):
            return random.randint(0, self.action_space-1)

        action = torch.argmax(self.network.forward(state)).item()
        return action

    def remember(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        batch = random.sample(self.replay_memory, min(len(self.replay_memory), batch_size))
        for state, action, reward, next_state, done in batch:
            y = self.network.forward(state)
            y[action] = reward if done else reward + self.gamma * torch.max(self.target_network.forward(next_state)).item()
            x_batch.append(state)
            y_batch.append(y)

        self.network.train(torch.Tensor(x_batch), torch.stack(y_batch))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update(self):
        for target_p, p, in zip(self.target_network.parameters(), self.network.parameters()):
            target_p.data.copy_(self.tau * p + (1 - self.tau) * target_p)
            # print('yeah')

    
    def run(self, env, batch_size, solved_count=np.inf, max_episodes=10000):
        rewards = collections.deque(maxlen=100)
        for _ in range(max_episodes):
            done =  False
            state = env.reset()
            total_rewards = 0
            for step in range(196):
                action = self.act(state)
                next_state, reward, done, info = env.step(action)
                reward = reward if not done else -10
                total_rewards += reward
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if len(self.replay_memory) >= batch_size:
                    self.replay(batch_size)

                if done:
                    if (_ % 50) == 0:
                        print("Episode {}, Score={}".format(_, total_rewards))
                    break
                
                step += 1

                rewards.append(total_rewards)

            if sum(rewards) / len(rewards) >= solved_count:
                print("Solved, episode {}".format(_))

            self.update()