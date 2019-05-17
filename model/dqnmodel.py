import numpy as np
import torch
import torch.nn as nn 

class DQN(nn.Module):
    def __init__(self, env):
        super(DQN, self).__init__()
        self.env = env  
        
        self.layers = nn.Sequential(
            nn.Linear(self.env.observation_space.shape[0], 128),  
            nn.ReLU(),  
            nn.Linear(128, 128),  
            nn.ReLU(),            
            nn.Linear(128, self.env.action_space.n)   
        )                                  
                    
    def forward(self, x):  
        return self.layers(x)
    
    def select_action(self, state, epsilon): 
        if np.random.random() > epsilon:
            q_value = self.forward(state)  
            _, action  = q_value.max(0) 
            action = action.item() 
        else:
            action = self.env.action_space.sample()  
        return action    