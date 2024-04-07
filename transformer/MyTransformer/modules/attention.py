import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# attention for one head
def attention(q,k,v,mask=None,dropout=None):
    dimension = q.size()[-1]
    score = torch.matmul(q,k.transpose(-2,-1)) / math.sqrt(dimension)
    if mask is not None:
        score = score.masked_fill(mask==0,1e-9)
    attention_score = F.softmax(score,dim=-1)
    if dropout is not None:
        attention_score = dropout(attention_score)
    return torch.matmul(attention_score,v) , attention_score

class MultiHeadAttention(nn.Module):
    def __init__(self,embedding_d:int,head_n:int,dropout:float=0.1):
        super(MultiHeadAttention,self).__init__()
        assert embedding_d % head_n == 0, f"you should give valid head_n, rather than {head_n}"
        self.head_n = head_n
        self.embeding_d = embedding_d
        self.q_w, self.k_w, self.v_w, self.out_w = [nn.Linear(embedding_d,embedding_d,bias=False
                                                    ) for _ in range(4)]
        self.attention_score = None
        self.dropout = nn.Dropout(dropout)

    def forward(self,q,k,v,mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) # [n_batch x seq x d x d] -> [n_batch x 1 x seq x d x d]
        
        batch_size = q.size()[0]

        # [n_batch x seq x d ] -> [n_batch x head_n x seq x d/head_n]
        q,k,v = [ linear(x).view(batch_size,-1,self.head_n,self.embeding_d).transpose(1,2)  for linear ,x in zip([self.q_w,self.k_w,self.v_w],[q,k,v])]

        # x: [n_batch x head_n x seq x d/head_n]
        x, self.attention_score = attention(q,k,v,mask,self.dropout)
        # x: [n_batch x seq x d]
        x = x.transpose(1,2).contiguous().view(batch_size,-1,self.embeding_d)

        return self.out_w(x)

class AttentionLayer(nn.Module):
    # add layernorm + MultiheadAttetnion together
    def __init__(self,embedding_d:int,head_n:int,dropout:float=0.1):
        super(AttentionLayer,self).__init__()
        self.multihead_attention = MultiHeadAttention(embedding_d,head_n,dropout)
        self.norm = nn.Layernorm(embedding_d)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,q,k,v,mask):
        q_norm, k_norm, v_norm = [self.norm(elem) for elem in [q,k,v]]
        output = self.multihead_attention(q_norm,k_norm,v_norm,mask)
        return q + self.dropout(output)