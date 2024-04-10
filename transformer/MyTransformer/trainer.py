import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from tqdm import tqdm
import os
import json
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from MyTransformer.modules.transformer import Transformer
from MyTransformer.dataset.tokenizer import DigitTokenizer,CustomDataset

def generate_encoder_mask(input_seq,pad_value):
    return (input_seq!=pad_value).long().unsqueeze(1).unsqueeze(2)

def generate_decoder_mask(output_seq,pad_value):
    output_pad_mask = (output_seq!=pad_value).unsqueeze(1).unsqueeze(3)
    output_seq_len = output_seq.shape[1]
    output_tril_mask = torch.tril(torch.ones(output_seq_len,output_seq_len)).type(torch.ByteTensor)
    output_mask = output_pad_mask & output_tril_mask
    return output_mask

class Trainer():

    def __init__(self,transformer,dataloader,validloader,tokenizer,save_path) -> None:
        # add the model
        self.model = transformer
        # define optimizer
        self.opt = torch.optim.Adam(params=transformer.parameters(), lr=2e-4, betas=(0.9, 0.98), eps=1e-9)
        # add the dataset
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.save_path = save_path
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_value)
        self.init_train_step()
        if save_path is not None:
            # self.writer = SummaryWriter(os.path.join(save_path,"result"))
            self.writer = None
        else:
            self.writer = None
        self.validloader = validloader

    def init_train_step(self):
        self._train_step = 0
        self.accuracy_list = []
        self.train_loss_list = []
        self.valid_loss_list = []
    
    def run(self,epoch_num,frequency=5):
        for i in tqdm(range(epoch_num)):
            # train
            for batch in self.dataloader:
                self.update_one_batch(batch,True)
            # test
            if (i+1)%frequency==0:
                for test_batch in self.validloader:
                    self.update_one_batch(test_batch,False)
                    self.accuracy_list.append(self.test(test_batch[0]))
        with open(os.path.join(self.save_path,"result.json"),"w") as f:
            json.dump({"test_accuracy_list":self.accuracy_list,
                       "train_loss_list":self.train_loss_list,
                       "valid_loss_list":self.valid_loss_list},f)

    def update_one_batch(self,batch,train=True):
        self.opt.zero_grad()
        if train:
            self.model.train()
        else:
            self.model.eval()
        input_texts, _, input_seqs_padded, output_seqs_padded = batch[:4] 
        input_mask = generate_encoder_mask(input_seqs_padded,self.tokenizer.pad_value)
        output_mask = generate_decoder_mask(output_seqs_padded[:,:-1],self.tokenizer.pad_value)
        desired_output = output_seqs_padded[:,:-1].contiguous().view(-1)
        # inference
        transformer_output = self.model(input_seqs_padded,output_seqs_padded[:,:-1],output_mask,input_mask)
        logit_output = self.model.generator(transformer_output)
        logit_output = logit_output.view(-1,logit_output.shape[-1])

        # print("desired shape",desired_output.shape)
        # print("logit_output_shape: ",logit_output.shape)
        # print("start loss")
        loss = self.criterion(logit_output,desired_output)
        # print("start backward")
        if train:
            loss.backward()
        # print("start step")
            self.opt.step()
        loss = float(loss.detach().numpy())
        if train:
            self.train_loss_list.append(loss)
        else:
            self.valid_loss_list.append(loss)

    def inference(self,input_text:str):
        max_length = 10
        if input_text[-1]!=self.tokenizer.eos:
            input_text+=self.tokenizer.eos
        input_seqs = [self.tokenizer.tokenize(input_text) ]
        input_tensor = torch.tensor(input_seqs)
        self.model.eval()
        with torch.no_grad(): 
            encoder_output = self.model.encoder(input_tensor)

            output_txt = self.tokenizer.bos
            for _ in range(max_length):
                # greedy search
                output_tensor = torch.tensor([self.tokenizer.tokenize(output_txt)])
                decoder_output = self.model.decoder(output_tensor,encoder_output)
                logit_output = self.model.generator(decoder_output)[:,-1,:].view(-1).detach().numpy()
                output_index = np.argmax(logit_output)
                one_output_txt = self.tokenizer.detokenize([output_index])[0]
                output_txt += one_output_txt
                if one_output_txt == self.tokenizer.eos:
                    return output_txt
            return output_txt

    def test(self,input_list):
        accuracy_num = 0
        for input_text in input_list:
            output = self.inference(input_text)
            accuracy_num +=int(output[1:-1] == input_text[:-1][::-1])
        return accuracy_num/len(input_list)

    def save_checkpoint(self,file_name,save_path=None):
        if save_path is None:
            save_path = self.save_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        checkpoint = {
            "model": self.model.state_dict(),
            "train_step":self._train_step,
        }
        self.model.save(save_path)
        torch.save(checkpoint,os.path.join(os.path.join(save_path,file_name+".pth.tar")))
    
    @staticmethod
    def load_checkpoint(checkpoint_file,dataloader,valid_dataloader,tokenizer):
        if not os.path.exists(checkpoint_file):
            return None
        file_path = os.path.dirname(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        with open(os.path.join(file_path,"model_parameters.json"),"r") as f:
            model_parameters = json.load(f)
        model = Transformer(**model_parameters)
        model.load_state_dict(checkpoint["model"])
        trainer = Trainer(model,dataloader,valid_dataloader,tokenizer,file_path)
        trainer._train_step = checkpoint["train_step"]
        return trainer

if __name__=="__main__":
    BATCH_SIZE = 500
    EMBEDDING_D = 128
    HEAD_N = 2
    LAYER_N = 3

    with open("./dataset/dataset1.txt") as f:
        data_inputs = f.readlines()
    with open("./dataset/valid_dataset.txt") as f:
        valid_data_inputs = f.readlines()
    data_inputs = [line[:-1] for line in data_inputs]
    valid_data_inputs = [line[:-1] for line in valid_data_inputs]
    tokenizer = DigitTokenizer()
    vocab_size = tokenizer.max_length
    dataset = CustomDataset(data_inputs,tokenizer)
    valid_dataset = CustomDataset(valid_data_inputs,tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=valid_dataset.collate_fn)
    model = Transformer(embedding_d=EMBEDDING_D,ffn_d=EMBEDDING_D*2,head_n=HEAD_N,layer_n=LAYER_N,vocab_size=vocab_size,dropout=0.1)

    trainer = Trainer(model,dataloader,valid_dataloader,tokenizer,"./result")
    trainer.run(20)
    trainer.save_checkpoint("checkpoint")