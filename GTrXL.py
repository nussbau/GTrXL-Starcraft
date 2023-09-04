import torch.nn as nn
import torch
from config import *
from math import sqrt, log


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class Attention_Module(nn.Module):
    
    def __init__(self, in_size, head_dim):
        super(Attention_Module, self).__init__()
        self.Q = nn.Linear(in_size, head_dim)
        self.K = nn.Linear(in_size, head_dim)
        self.V = nn.Linear(in_size, head_dim)
        self.R = nn.Linear(in_size, head_dim)
        self.u = nn.Parameter(torch.zeros(head_dim))
        self.v = nn.Parameter(torch.zeros(head_dim))

        self.pos_encoding = None
        self.mask = None

    def forward(self, inp_data):
        if self.pos_encoding is None:
            self.pos_encoding = positionalencoding1d(inp_data.shape[-1], inp_data.shape[0]).unsqueeze(1).to('cuda')
            

        # Query
        q = self.Q(inp_data[:inp_data.shape[0]//2])

        # Key
        k = self.K(inp_data)

        # Value
        v = self.V(inp_data)

        r = self.R(self.pos_encoding)

        uk = torch.einsum('thd, qhm -> htq', self.u.repeat((inp_data.shape[0]//2, 1, 1)), k)
        vr = torch.einsum('thd, qhm -> htq', self.v.repeat((inp_data.shape[0]//2, 1, 1)), r)

        qk_t = torch.einsum('thd, mhd -> htm', q, k) + torch.einsum('thd, mhd -> htm', q, r) + uk + vr

        # Masking for softmax
        qk_t[:, :, :inp_data.shape[0]//2][torch.triu(torch.ones((inp_data.shape[1], memory_size, memory_size), dtype=bool), diagonal=1)] = -1e20
        qk_t = qk_t.transpose(1, 0)

        attention_score = nn.functional.softmax(qk_t, dim=-1)

        attention_value = torch.einsum('ijk, kjl -> ijl', attention_score, v)

        return attention_value

class Multi_Headed_Attention(nn.Module):

    def __init__(self, num_heads, embedding_size, head_dim):
        super(Multi_Headed_Attention, self).__init__()
        self.heads = nn.ModuleList([Attention_Module(embedding_size, head_dim) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads*head_dim, embedding_size)
        self.norm = nn.LayerNorm(embedding_size)

    def forward(self, inp_data):
        x = torch.cat([attention(inp_data) for attention in self.heads], axis=-1)
        x = self.linear(x) + inp_data[:inp_data.shape[0]//2]
        x = self.norm(x)
        return x


class Gated_Reccurent_Unit(nn.Module):
    
    def __init__(self, input_size):
        super(Gated_Reccurent_Unit, self).__init__()
        self.Ur = nn.Linear(input_size, input_size, bias=False)
        self.Wr = nn.Linear(input_size, input_size, bias=False)
        self.Br = nn.Parameter(torch.zeros((1, input_size)))

        self.Uz = nn.Linear(input_size, input_size, bias=False)
        self.Wz = nn.Linear(input_size, input_size, bias=False)
        self.Bz = nn.Parameter(torch.zeros((1, input_size)))

        self.Uh = nn.Linear(input_size, input_size, bias=False)
        self.Wh = nn.Linear(input_size, input_size, bias=False)
        self.Bh = nn.Parameter(torch.zeros((1, input_size)))

        self.sigmoid = nn.Sigmoid()
        self.tan = nn.Tanh()

    def forward(self, inp_data, memory):
        reset_value = self.sigmoid(self.Wr(inp_data) + self.Ur(memory) + self.Br)
        update_value = self.sigmoid(self.Wz(inp_data) + self.Uz(memory) + self.Bz)

        hhat = self.tan(self.Wh(inp_data) + self.Uh(reset_value * memory) + self.Bh)
        h = ((1-update_value) * memory + update_value * hhat)
        
        return h
    
class TransformerBlock(nn.Module):

    def __init__(self, num_heads, embedding_size, head_dim, memory_length):
        super(TransformerBlock, self).__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_size),
            Multi_Headed_Attention(num_heads, embedding_size, head_dim),
            nn.ReLU()
        )

        self.gru1 = Gated_Reccurent_Unit(embedding_size)

        self.tail = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Conv2d(in_channels=memory_length, out_channels=memory_length, kernel_size=1, stride=1), 
            nn.ReLU()
        )

        self.gru2 = Gated_Reccurent_Unit(embedding_size)

    def forward(self, inp_data, memory_embedding):
        x = torch.cat([inp_data, memory_embedding], axis=0)
        x = self.head(x)
        post_gru = self.gru1(x, inp_data)
        x = self.tail(post_gru)
        x = self.gru2(x, post_gru)
        return x



class GTrXL(nn.Module):
    
    def __init__(self, num_layers, head_dim, num_heads, embedding_size, memory_length):
        super(GTrXL, self).__init__()
        self.transformer = nn.ModuleList([TransformerBlock(head_dim=head_dim, num_heads=num_heads, embedding_size=embedding_size, memory_length=memory_length) for _ in range(num_layers)])

    def forward(self, inp_data, memory_embedding):

        new_memories = []
        for i, block in enumerate(self.transformer):
            inp_data = block(inp_data, memory_embedding[i])
            new_memories.append(inp_data)

        new_memories = torch.stack(new_memories).detach()

        return inp_data, new_memories
    

