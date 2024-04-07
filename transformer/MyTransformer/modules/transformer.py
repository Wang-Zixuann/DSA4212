import torch.nn as nn
import torch.nn.functional as F
from MyTransformer.modules.attention import MultiHeadAttention
from MyTransformer.modules.embedding_and_encoding import Embedding, PositionEncoding
from MyTransformer.modules.ffn import FFN

class EncoderLayer(nn.Module):
    def __init__(self,embedding_d,ffn_d,head_n,dropout=0.1):
        super(EncoderLayer,self).__init__()
        layers_list = []
        layers_list.append(MultiHeadAttention(embedding_d,
                                                  head_n,
                                                  dropout))
        layers_list.append(FFN(embedding_d,ffn_d,dropout))
        self.layers = nn.ModuleList(layers_list)
    
    def forward(self,input,mask=None):
        output = self.layers[0](input,input,input,mask)
        return self.layers[1](output)

class DecoderLayer(nn.Module):
    def __init__(self,embedding_d,ffn_d,head_n,dropout=0.1):
        super(DecoderLayer,self).__init__()
        layers_list = []
        for _ in range(2):
            layers_list.append(MultiHeadAttention(embedding_d,
                                                  head_n,
                                                  dropout))
        layers_list.append(FFN(embedding_d,ffn_d,dropout))
        self.layers = nn.ModuleList(layers_list)
    
    def forward(self,input,encoder_output,decoder_mask=None,encoder_mask=None):
        output = self.layers[0](input,input,input,decoder_mask)
        output = self.layers[1](output,encoder_output,encoder_output,encoder_mask)
        output = self.layers[2](output)
        return output

class Generator(nn.Module):
    def __init__(self,embedding_d,vocab_size):
        super(Generator,self).__init__()
        self.proj = nn.Linear(embedding_d,vocab_size)
    
    def forward(self,input):
        return F.log_softmax(self.proj(input),dim=-1)

class Transformer(nn.Module):
    def __init__(self,embedding_d,ffn_d,head_n,layer_n,vocab_size,dropout=0.1):
        super(Transformer,self).__init__()
        self.encoder_list = nn.ModuleList([EncoderLayer(embedding_d,ffn_d,head_n,dropout) for _ in range(layer_n)])
        self.decoder_list = nn.ModuleList([DecoderLayer(embedding_d,ffn_d,head_n,dropout) for _ in range(layer_n)])
        self.embed = nn.Sequential(Embedding(vocab_size,embedding_d),
                                   PositionEncoding(embedding_d,vocab_size,dropout))
        # self.embed = Embedding(vocab_size,embedding_d)
        self.generator = Generator(embedding_d,vocab_size)

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def encoder(self,input,mask=None):
        input = self.embed(input)
        # print("embedding output for encoder: ",input)
        print("embedding output size for encoder: ",input.size())
        for layer in self.encoder_list:
            input = layer(input,mask)
        return input
    
    def decoder(self,input,encoder_output,
                decoder_mask=None,encoder_mask=None):
        input = self.embed(input)
        for layer in self.decoder_list:
            input = layer(input,encoder_output,decoder_mask,encoder_mask)
        return input
    
    def forward(self,encoder_intput,decoder_input,
                decoder_mask=None,encoder_mask=None):
        encoder_output = self.encoder(encoder_intput,encoder_mask)
        return self.decoder(decoder_input,encoder_output,decoder_mask,encoder_mask)
