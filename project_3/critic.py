import torch
import numpy as np



class Modul(torch.nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 1)
        )
        

    def forward(self, x):
        return self.net(x)

    
    
class Critic:
    
    def __init__(self, num_inputs):
        self.discount_factor = 0.99
        
        self.modul = Modul(num_inputs)
        self.optimizer = torch.optim.SGD(self.modul.parameters(), lr=0.001)
        print(f'{self.modul}\n')
        
        
    def compute_delta(self, reward, state, new_state):
        reward = torch.tensor(reward, dtype=torch.float32)
        state = torch.tensor(state, dtype=torch.float32)
        new_state = torch.tensor(new_state, dtype=torch.float32)
        
        loss = torch.nn.MSELoss()
        
        next_val = torch.squeeze(self.modul.forward(new_state).detach())
        target = reward + self.discount_factor*next_val
        output = torch.squeeze(self.modul.forward(state))

        self.optimizer.zero_grad()
        loss_critic = loss(output, target)
        torch.nn.utils.clip_grad_norm_(self.modul.parameters(), 1.)

        loss_critic.backward()
        self.optimizer.step()
        
        delta = target-output
        return delta.item(), loss_critic.item()
    

