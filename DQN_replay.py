from replay import PrioritizedMemory
import torch
import numpy as np
import random
import gym
from networks import BasicDNN
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

class DQLearner:

    def __init__(self, state_size, action_size, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01, memory_length=10000, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.gamma = gamma

        self.network = BasicDNN(self.state_size, self.action_size)

        self.memory_length = memory_length
        self.memory = PrioritizedMemory(self.memory_length, alpha=0.6)

        self.steps_taken = 0
        self.replay_count = 0

    def epsilon_update(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_action(self, state):
        q_pred = self.network.forward(state)
        action = random.randint(
            0, self.action_size-1) if self.epsilon <= random.random() else np.argmax(q_pred)
        return (action, q_pred)

    def get_target(self, reward, next_state, done):
        return reward if done else reward + self.gamma * np.max(self.network.forward(next_state))

    def remember(self, state, action, reward, next_state, done, error):
        self.memory.insert(error, (state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        batch_data = [list(self.memory.retrieve()) for i in range(batch_size)] # List of [(state, action, reward, next_state, done), index]
        samples = []
        target_cache = []
        indices = []
        for sample in batch_data:
            state, action, reward, next_state, done = sample[0]
            target = self.network.forward(state)
            target_val = self.get_target(reward, next_state, done)
            target[action] = target_val
            samples.append((state, target))
            indices.append(sample[1])
            target_cache.append(target_val)
        
        x = np.stack([state for state, target in samples])
        y = np.stack([target for state, target in samples])
        loss, q_pred = self.network.train(x, y)
        writer.add_scalar('Loss', loss, self.replay_count)
        self.replay_count += 1
        # Update Priority
        new_error = np.linalg.norm(q_pred - np.expand_dims(np.array(target_cache), 1), axis=1)
        for i in range(new_error.shape[0]):
            self.memory.update(indices[i], new_error[i])

        self.epsilon_update()

    def run(self, env, episodes, max_steps):
        self.steps_taken = 0
        for e in range(episodes):
            done = False
            state = env.reset()
            cummulative_reward = 0
            for s in range(max_steps):
                action, q_pred = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                error = np.abs(reward) if not self.memory.is_full() else np.linalg.norm(
                    self.get_target(reward, next_state, done) - q_pred[action])

                self.remember(state, action, reward, next_state, done, error)

                self.steps_taken += 1
                cummulative_reward += reward

                state = next_state

                if self.memory.is_full():
                    self.replay()

                if done:
                    break
            
            writer.add_scalar('Reward', cummulative_reward, e)

            if not e % 10:
                print("Episode: {}/{}, Reward: {}".format(e, episodes, cummulative_reward))


if __name__ == '__main__':
    torch.random.manual_seed(3)
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    action_space = 2
    state_space = 4
    env = gym.make('CartPole-v0')
    agent = DQLearner(state_space, action_space)
    agent.run(env, 5000, 200)
    