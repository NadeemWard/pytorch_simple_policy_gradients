import torch
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Gaussian_Policy(nn.Module):
    '''
    Gaussian policy that consists of a neural network with 1 hidden layer that
    outputs mean and log std dev (the params) of a gaussian policy
    '''

    def __init__(self, num_inputs, hidden_size, action_space):

       	super(Gaussian_Policy, self).__init__()

        self.action_space = action_space
        num_outputs = action_space.shape[0] # the number of output actions

        self.linear = nn.Linear(num_inputs, hidden_size)
        self.mean = nn.Linear(hidden_size, num_outputs)
        self.log_std = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):

        # forward pass of NN
        x = inputs
        x = F.relu(self.linear(x))

        mean = self.mean(x)
        log_std = self.log_std(x) # if more than one action this will give you the diagonal elements of a diagonal covariance matrix
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX) # We limit the variance by forcing within a range of -2,20
        std = log_std.exp()

        return mean, std

class ValueNetwork(nn.Module):
    '''
    Value network V(s_t) = E[G_t | s_t] to use as a baseline in the reinforce
    update. This a Neural Net with 1 hidden layer
    '''

    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, state):

        x = F.relu(self.linear1(state))
        x = self.linear2(x)

        return x


