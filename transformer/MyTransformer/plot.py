import matplotlib.pyplot as plt
import json


if __name__=="__main__":
    with open("./result/result.json","r") as f:
        result = json.load(f)
    
    train_loss = result["train_loss_list"]
    plt.plot([i for i in range(len(train_loss))],train_loss)
    plt.show()