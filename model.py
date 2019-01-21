import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(state_size, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, action_size))

    def forward(self, state):
            """Build a network that maps state -> action values."""
            return self.net(state)

class Dueling_QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(Dueling_QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.net = nn.Sequential(nn.Linear(state_size, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU())
        self.V = nn.Linear(32,action_size)
        self.A = nn.Linear(32,1)

    def forward(self, state):
            """Build a network that maps state -> action values."""
            net = self.net(state)
            A = self.A(net)
            V = self.V(net)
            return V + (A - torch.mean(A))


