import 
import Networks
import torch
import gym

torch.random.manual_seed(2)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

action_space = 2
state_space = 4

env = gym.make('CartPole-v1')
network = Networks.BasicDNN(state_space, action_space)
dqn = DQN.DQN(network, action_space)

dqn.run(env, 32, solved_count=195)
