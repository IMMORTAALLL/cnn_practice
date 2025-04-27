import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.net=nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,stride=1,padding=2),nn.Tanh(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5), nn.Tanh(),
            nn.Flatten(),
            nn.Linear(120, 84), nn.Tanh(),
            nn.Linear(84, 10)
        )
    def forward(self,x):
        y=self.net(x)
        return y
