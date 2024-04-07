import torch
from torch.utils.data import Dataset, DataLoader
from MyTransformer.modules.transformer import Transformer
from MyTransformer.dataset.tokenizer import DigitTokenizer,CustomDataset

class Trainer():

    def __init__(self,transformer,dataset) -> None:
        # add the model
        self.model = transformer
        # define optimizer
        self.opt = torch.optim.Adam(transformer.parameters(), lr=2e-4, betas=(0.9, 0.98), eps=1e-9)
        # add the dataset
        self.dataset = dataset

    def run(self,epoch_num):
        pass

    def update_one_epoch(self):
        pass

    def calculate_loss(self,inputs,outputs):
        pass

    def inference(self):
        pass

    def save(self):
        pass

if __name__=="__main__":
    with open("./dataset/dataset1.txt") as f:
        data_inputs = f.readlines()
    data_inputs = [line[:-1] for line in data_inputs]
    print(data_inputs[0])
    tokenizer = DigitTokenizer()
    dataset = CustomDataset(data_inputs,tokenizer)
    dataloader = DataLoader(dataset, batch_size=3, collate_fn=dataset.collate_fn)
    i = 0
    model = Transformer(embedding_d=20,ffn_d=20,head_n=2,layer_n=2,vocab_size=20,dropout=0.1)
    for batch in dataloader:
        input, ouput = batch
        print("input:  ",input)
        model_ouput = model(input,ouput)
        softmax_ouput = model.generator(model_ouput)
        print("softmax output: ", softmax_ouput.shape)
        break
    