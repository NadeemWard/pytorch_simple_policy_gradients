from policy import Softmax_Policy
from policy import ValueNetwork
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import ipdb

class Actor_Critic_discrete:
    '''
    Implementation of the basic one step actor-critic algorithm for discrete
    action spaces with baseline
    '''
    def __init__(self, num_inputs, hidden_size, action_space, lr_pi = 3e-4,\
                 lr_vf = 1e-3, baseline = False, gamma = 0.99, train_v_iters = 1):

        self.gamma = gamma
        self.action_space = action_space
        self.policy = Softmax_Policy(num_inputs, hidden_size, action_space)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr = lr_pi)
        self.baseline = baseline
        self.train_v_iters = train_v_iters # how many times you want loop training of value network
        self.critic = ValueNetwork(num_inputs, hidden_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = lr_vf)

    def select_action(self,state):

        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)

        return action.item(), log_prob


    def train(self, trajectory):
        '''
        Training is done through a boostrap estimate using a value network.
        trajectory: a list of the form [( state , action , lnP(a_t|s_t), reward ), ...
        ]'''

        log_probs = [item[2] for item in trajectory]
        rewards = [item[3] for item in trajectory]
        states = [item[0] for item in trajectory]
        actions = [item[1] for item in trajectory]

        value_estimates = []
        for state in states:
            state = torch.from_numpy(state).float().unsqueeze(0) # just to make it a Tensor obj
            value_estimates.append( self.critic(state) )

        next_state_estimates = []
        for indx in range(1, len(value_estimates) ):
            next_state_estimates.append(np.asscalar(value_estimates[indx].detach().numpy()))

        next_state_estimates.append(0)
        value_estimates = torch.stack(value_estimates).squeeze()

        boostrap_estimate = []
        for indx in range(len(rewards)):
            G = rewards[indx] + self.gamma*next_state_estimates[indx]
            boostrap_estimate.append(G)
        boostrap_estimate = torch.Tensor(boostrap_estimate)

#        ipdb.set_trace()
        # train the Value Network and calculate Advantage
        if self.baseline:

            # calculate advantage
            advantage = []
            for value, R in zip(value_estimates, boostrap_estimate):
                advantage.append(R - value)

            advantage = torch.Tensor(advantage)

            # caluclate policy loss
            policy_loss = []
            for log_prob, adv in zip(log_probs, advantage):
                policy_loss.append( - log_prob * adv)

        else:
            policy_loss = []
            for log_prob, R in zip(log_probs, boostrap_estimate):
                policy_loss.append( - log_prob * R)

        # update value network
        for _ in range(self.train_v_iters):

            v_loss = F.mse_loss(value_estimates, boostrap_estimate)
            # update the weights
            self.critic_optimizer.zero_grad()
            v_loss.backward()
            self.critic_optimizer.step()

        # update policy network
        policy_loss = torch.stack( policy_loss ).sum()
        # update policy weights
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()


        return policy_loss, v_loss
