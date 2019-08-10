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

    def forward(self, x):
        x = torch.Tensor(x).cuda()
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x.numpy()

    def train_forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

    def train(self, x, y):
        x = torch.Tensor(x).cuda()
        y = torch.Tensor(y).cuda()
        y_hat = self.train_forward(x)
        loss = self.loss(y_hat, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return (loss.item(), y_hat.numpy())