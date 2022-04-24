import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def to_cuda(elements):
    if not torch.cuda.is_available():
        return elements
    if isinstance(elements, tuple) or isinstance(elements, list):
        return [x.cuda() for x in elements]
    return elements.cuda()



class nn_Critic(nn.Module):

    def __init__(self, num_inputs, layers):
        super(nn_Critic, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.num_layers = len(layers)
        
        self.fc_layers.append(nn.Linear(num_inputs, layers[0]))
        
        for i in range(self.num_layers-1):
            self.fc_layers.append(nn.Linear(layers[i], layers[i+1]))


    def forward(self, state):
        m = nn.ReLU()
        for i in range(self.num_layers-1):
            out = self.fc_layers[i](state)
            state = m(out)
        output = self.fc_layers[-1](state)
        return output
    
    
    
class Critic:
    
    def __init__(self, num_inputs, layers=[15, 20, 30, 5, 1], learning_rate=0.001, discount_factor=0.95):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.nn_critic = True
        
        self.modul = to_cuda(nn_Critic(num_inputs, layers))
        self.optimizer = torch.optim.SGD(self.modul.parameters(), lr=learning_rate)
        #self.optimizer = torch.optim.Adam(self.modul.parameters(), lr=learning_rate)
        print(f'{self.modul}\n')
        
        
    def prepare_for_epoch(self):
        pass
        
        
    def compute_delta(self, reward, state, new_state, verbal=False):
        reward = torch.tensor(reward, dtype=torch.float32)
        state = torch.tensor(state, dtype=torch.float32)
        new_state = torch.tensor(new_state, dtype=torch.float32)
        
        loss = nn.MSELoss()
        
        next_val = torch.squeeze(self.modul.forward(new_state).detach())
        target = reward + self.discount_factor*next_val
        output = torch.squeeze(self.modul.forward(state))

        self.optimizer.zero_grad()
        loss_critic = loss(output, target)
        torch.nn.utils.clip_grad_norm_(self.modul.parameters(), 1.)

        loss_critic.backward()
        self.optimizer.step()
        
        
        if verbal:
            print(f'next_val: {next_val:1.4f} target: {target:1.4f} output: {output:1.4f} Loss: {loss_critic.item():1.4f}')
        
        delta = target-output
        return delta.item(), loss_critic.item()
    

