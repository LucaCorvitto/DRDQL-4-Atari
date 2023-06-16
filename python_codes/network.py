import torch
import torch.nn as nn

class DQN(nn.Module):
    
    def __init__(self, in_channels, num_actions):
        super().__init__()
        
        network = [
            torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7,512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        ]
        
        self.network = nn.Sequential(*network)
    
    def forward(self, x):
        actions = self.network(x)
        return actions

class Agent(nn.Module):
    
    def __init__(self, in_channels, num_actions, epsilon):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.network = DQN(in_channels, num_actions)
        
        self.eps = epsilon
    
    def forward(self, x):
        actions = self.network(x)
        return actions
    
    def e_greedy(self, x):
        
        actions = self.forward(x)
        
        greedy = torch.rand(1)
        if self.eps < greedy:
            return torch.argmax(actions)
        else:
            return (torch.rand(1) * self.num_actions).type('torch.LongTensor')[0] 
    
    def set_epsilon(self, epsilon):
        self.eps = epsilon