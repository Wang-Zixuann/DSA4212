import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self,emnbeding_d:int,ffn_d:int,dropout:float=0.1):
        super(FFN,self).__init__()
        self.linear1 = nn.Linear(emnbeding_d,ffn_d)
        self.linear2 = nn.Linear(ffn_d,emnbeding_d)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))