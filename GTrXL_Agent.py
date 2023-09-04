import numpy as np
import torch.nn as nn
from GTrXL import GTrXL
import torch
from torch.distributions.categorical import Categorical

def init_weights(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

class Agent(nn.Module):

    def __init__(self, n_input: int, n_output: int, embedding_dim: int, memory_length: int) -> None:
        super(Agent, self).__init__()

        assert embedding_dim % 2 == 0

        self.conv_stack = nn.Sequential(
            init_weights(nn.Conv2d(8, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            init_weights(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            init_weights(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            init_weights(nn.Linear(9216, embedding_dim//2)),
            nn.ReLU()
        ) 

        self.embedding = nn.Sequential(
            init_weights(nn.Linear(n_input, embedding_dim//2)),
            nn.ReLU(),
        )     

        self.transformer = GTrXL(num_layers=4, head_dim=64, num_heads=4, embedding_size=embedding_dim, memory_length=memory_length)
        self.transformer_tail = nn.Sequential(   
            init_weights(nn.Conv2d(memory_length, 1, kernel_size=1, stride=1)),
            nn.Tanh()
        )

        self.actor_stack = nn.Sequential(
            init_weights(nn.Linear(embedding_dim, embedding_dim//2)),
            nn.Tanh(),
            init_weights(nn.Linear(embedding_dim//2, n_output)),
            nn.Softmax(dim=-1)
        )       

        self.critic_stack = nn.Sequential(  
            init_weights(nn.Linear(embedding_dim, embedding_dim//2)),
            nn.Tanh(),
            init_weights(nn.Linear(embedding_dim//2, 1))
        )

    
    def forward(self, inp_data, map, current_sequence, old_memories):
        state_embedding = torch.cat((self.embedding(inp_data), self.conv_stack(map)), dim=1).unsqueeze(0)

        x = torch.cat((state_embedding, current_sequence))

        x, new_memories = self.transformer(x, old_memories)

        x = self.transformer_tail(x)

        action_probs = self.actor_stack(x)

        return Categorical(action_probs), state_embedding, new_memories
    

    def forward_critic(self, inp_data, map, current_sequence, old_memories):
        state_embedding = torch.cat((self.embedding(inp_data), self.conv_stack(map)), dim=1).unsqueeze(0)

        x = torch.cat((state_embedding, current_sequence))

        x, _ = self.transformer(x, old_memories)

        x = self.transformer_tail(x)

        return self.critic_stack(x)



class Memory():

    def __init__(self, batch_size) -> None:
        self.memory = {
            'states': [],
            'maps': [],
            'memories': [],
            'old_memories': [],
            'probs': [],
            'values': [],
            'actions': [],
            'rewards': [],
            'dones': []
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
        

    def clear_memory(self):
        self.memory = {
            'states': [],
            'maps': [],
            'memories': [],
            'old_memories': [],
            'probs': [],
            'values': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }
       