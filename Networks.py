import torch 
import torch.nn as nn
import torch.nn.functional as F

class BasicDNN(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.l1 = nn.Linear(input_size, 12)
        self.l2 = nn.Linear(12, 12)
        self.l3 = nn.Linear(12,  output_size)

        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss = torch.nn.MSELoss()

    def forward(self, x, train=False):
        x = torch.Tensor(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

    def train(self, x, y):
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()