import math
import torch
import torch.nn as nn
from torch.autograd import Variable

class Embedding(nn.Module):
    def __init__(self,embedding_d:int,vocab_size:int):
        super(Embedding,self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_d)
        self.embedding_d = embedding_d
    
    def forward(self,x):
        # print("input of embedding shape: ",x.shape)
        return self.embed(x) * math.sqrt(self.embedding_d)
    
class PositionEncoding(nn.Module):
    def __init__(self,embedding_d,vocab_size,dropout=0.1):
        super(PositionEncoding,self).__init__()
        self.vocab_size = vocab_size
        self.embedding_d = embedding_d
        self.dropout = nn.Dropout(dropout)
        # sin
        position_encoding = torch.zeros(vocab_size,embedding_d)
        position = torch.arange(0, vocab_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,embedding_d,2) *
                             -(math.log(1000)/embedding_d))
        position_encoding[:,0::2] = torch.sin(position*div_term)
        position_encoding[:,1::2] = torch.cos(position*div_term)
        position_encoding = position_encoding.unsqueeze(0)
        self.register_buffer("position_encoding",position_encoding)
    
    def forward(self,x):
        # print("x shape: ",x.shape)
        pe_value = Variable(self.position_encoding[:,:x.shape[1],:],
                         requires_grad=False)
        # print("position encoding shape: ",pe_value.shape)
        x = x + pe_value
        return self.dropout(x)