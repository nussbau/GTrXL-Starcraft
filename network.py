import torch.nn as nn
import numpy as np
import torch
from config import *
from torch.distributions.categorical import Categorical

def init_weights(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

class starcraft_network(nn.Module):

    def __init__(self, n_input: int, n_output: int) -> None:
        super(starcraft_network, self).__init__()
        self.first_linear = nn.Sequential(
            init_weights(nn.Linear(n_input, 512)),
            nn.ReLU(),
        )

        self.conv_stack = nn.Sequential(
            init_weights(nn.Conv2d(8, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            init_weights(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            init_weights(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            init_weights(nn.Linear(9216 ,512)),
            nn.ReLU()
        )      
        
        self.actor_stack = nn.Sequential(
            init_weights(nn.Linear(1024, 512)),
            nn.Tanh(),
            init_weights(nn.Linear(512, 256)),
            nn.Tanh(),        
            init_weights(nn.Linear(256, n_output), std=0.01),
            nn.Softmax(dim=1)
        )       

        self.critic_stack = nn.Sequential(
            init_weights(nn.Linear(1024, 512)),
            nn.Tanh(),
            init_weights(nn.Linear(512, 256)),
            nn.Tanh(),        
            init_weights(nn.Linear(256, 1), std=1),
        )
    
    def forward(self, inp_data, map):
        parameter_encoding = self.first_linear(inp_data)
        map_encoding = self.conv_stack(map)
        game_state_encoding = torch.cat((parameter_encoding, map_encoding), dim=1)
        action_probs = self.actor_stack(game_state_encoding)
        return Categorical(action_probs)
    
    def forward_critic(self, inp_data, map):
        parameter_encoding = self.first_linear(inp_data)
        map_encoding = self.conv_stack(map)
        game_state_encoding = torch.cat((parameter_encoding, map_encoding), dim=1)
        return self.critic_stack(game_state_encoding)


class Memory():

    def __init__(self, batch_size) -> None:
        self.memory = {
            'states': [],
            'maps': [],
            'probs': [],
            'values': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }

        self.last_memory = {
            'states': None,
            'maps': None,
            'probs': None,
            'values': None,
            'actions': None,
            'rewards': None,
            'dones': None
        }

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.memory['states'])
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int16)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return self.memory, batches
    
    def store_memory(self, mem: dict):
        for key in self.memory:
            self.memory[key].append(mem[key])
            self.last_memory[key] = mem[key]
        

    def clear_memory(self):
        self.memory = {
            'states': [],
            'maps': [],
            'probs': [],
            'values': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }