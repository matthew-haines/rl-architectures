import sys
import gym
import torch
import numpy as np
from networks import BasicDNN
import time

env = gym.make('CartPole-v0')

model = BasicDNN(4, 12, 2)
model.load_state_dict(torch.load(sys.argv[1], map_location=torch.device('cpu')))

while True:
    state = env.reset()
    for step in range(200):
        time.sleep(1/120)
        env.render()
        action = np.argmax(model.forward(state))
        state, reward, done, _ = env.step(action)
        
        if done:
            break