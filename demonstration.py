import sys
import gym
import torch
import numpy as np
from networks import BasicDNN
import time

env = gym.make('CartPole-v0')

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

model = BasicDNN(4, int(sys.argv[2]), 2)
model.load_state_dict(torch.load(sys.argv[1]))

while True:
    state = env.reset()
    for step in range(200):
        time.sleep(1/120)
        env.render()
        action = np.argmax(model.forward(state))
        state, reward, done, _ = env.step(action)
        
        if done:
            break