import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class DigitTokenizer:
    def __init__(self, max_length=11):
        self.max_length = max_length
        self.vocab = {str(i): i for i in range(10)}
        self.bos = "B"
        self.eos = "E"
        self.pad = "P"
        self.bos_value = 10
        self.eos_value = 11
        self.pad_value = 12     
        self.vocab[self.bos] = self.bos_value
        self.vocab[self.eos] = self.eos_value
        self.vocab[self.pad] = self.pad_value
        self.reverse_vocab = {i: str(i) for i in range(10)}
        self.reverse_vocab[self.bos_value] = self.bos
        self.reverse_vocab[self.eos_value] = self.eos
        self.reverse_vocab[self.pad_value] = self.pad
        self.max_length = len(self.vocab)

    def tokenize(self, text):
        # 将字符串转换为整数索引列表
        return [self.vocab[char] for char in text]

    def detokenize(self, tokens):
        # 将整数索引列表转换回字符串
        return ''.join(self.reverse_vocab[token] for token in tokens)

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq = self.data[idx]
        output_seq = self.tokenizer.bos+input_seq[::-1][1:]+self.tokenizer.eos  # Reverse the input sequence
        return input_seq, output_seq

    def collate_fn(self,batch):
        # input_seqs, output_seqs = zip(*batch)
        input_seqs_txt, output_seqs_txt = zip(*batch)
        input_seqs = [self.tokenizer.tokenize(seq) for seq in input_seqs_txt]
        output_seqs = [self.tokenizer.tokenize(seq) for seq in output_seqs_txt]
        # Pad input and output sequences to the maximum length in the batch
        input_seqs_padded = pad_sequence([torch.tensor(seq) for seq in input_seqs], batch_first=True, padding_value=self.tokenizer.pad_value)
        output_seqs_padded = pad_sequence([torch.tensor(seq) for seq in output_seqs], batch_first=True, padding_value=self.tokenizer.pad_value)

        return input_seqs_txt,output_seqs_txt, input_seqs_padded, output_seqs_padded
    
def gen_dataset(path,dataset_num=2*10e3):
    from random import randint
    import json
    MAX_LENGHT = 5
    # need another Ending character
    all_data_list = []
    for _ in range(int(dataset_num)): 
        size = randint(1,MAX_LENGHT)
        all_data_list.append("".join([str(randint(0,10)) for _ in range(size)])+"E")
    with open(path,"w") as f:
        f.writelines("\n".join(all_data_list))

if __name__ == "__main__":
    # with open("./dataset1.txt") as f:
    #     data = f.readlines()
    #     data = [elem[:-1] for elem in data]
    # tokenizer = DigitTokenizer()
    # # Create dataset
    # dataset = CustomDataset(data,tokenizer)

    # # Create data loader
    # batch_size = 3
    # dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)

    # Tokenize function
    # def tokenize_batch(batch):
    #     input_seqs, output_seqs = zip(*batch)
    #     tokenized_input_seqs = [simple_tokenizer(seq) for seq in input_seqs]
    #     tokenized_output_seqs = [simple_tokenizer(seq) for seq in output_seqs]
    #     return tokenized_input_seqs, tokenized_output_seqs

    # Example usage:
    # for batch in dataloader:
    #     print(batch)
    #     break
    gen_dataset("./valid_dataset.txt")
