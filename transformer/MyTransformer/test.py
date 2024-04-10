from torch.utils.data import DataLoader
from MyTransformer.trainer import Trainer
from MyTransformer.modules.transformer import Transformer
from MyTransformer.dataset.tokenizer import DigitTokenizer,CustomDataset

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
    trainer = Trainer.load_checkpoint("./result/checkpoint.pth.tar",dataloader,valid_dataloader,tokenizer)

    for input in ["0E","95E","8655E","3896E","743E","86368E","835E","704E"]:
        print(f"input: {input}  | output: {trainer.inference(input)}")