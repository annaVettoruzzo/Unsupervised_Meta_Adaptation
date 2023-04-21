import torch
import params
import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
from stateless import functional_call
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------
def zeros(nb):
    return torch.zeros(nb).to(DEVICE).long()


# -------------------------------------------------------------------
def ones(nb):
    return torch.ones(nb).to(DEVICE).long()

# -------------------------------------------------------------------
def write_in_file(file, file_directory):
    a_file = open(file_directory, "wb")
    pickle.dump(file, a_file)
    a_file.close()
    
# -------------------------------------------------------------------
def convert_into_int(array_name):
    class_label = 0
    for i in np.unique(array_name):
        idxs = np.where(array_name == i)
        array_name[idxs] = class_label
        class_label+=1
    return array_name.astype('int')

# -------------------------------------------------------------------
class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

# -------------------------------------------------------------------
def accuracy(pred, y_true):
    y_pred = pred.argmax(1).reshape(-1).cpu()
    y_true = y_true.reshape(-1).cpu()
    return accuracy_score(y_pred, y_true)

# -------------------------------------------------------------------
def init_data_match_dict(keys, vals, variation):
    data = {}
    for key in keys:
        data[key] = {}
        if variation:
            val_dim = vals
        else:
            val_dim = vals

        if params.dataset_name in ['RotatedMNIST']:
            data[key]['data'] = torch.rand((val_dim, params.img_c, params.img_w, params.img_h))

        data[key]['label'] = torch.rand((val_dim, 1))

    return data

# -------------------------------------------------------------------
def embedding_dist(x1, x2, tau=0.05, xent=False):
    if xent:
        # X1 denotes the batch of anchors while X2 denotes all the negative matches
        # Broadcasting to compute loss for each anchor over all the negative matches

        # Only implemnted if x1, x2 are 2 rank tensors
        if len(x1.shape) != 2 or len(x2.shape) != 2:
            print('Error: both should be rank 2 tensors for NT-Xent loss computation')

        # Normalizing each vector
        eps = 1e-8

        norm = x1.norm(dim=1)
        norm = norm.view(norm.shape[0], 1)
        temp = eps * torch.ones_like(norm)

        x1 = x1 / torch.max(norm, temp)

        norm = x2.norm(dim=1)
        norm = norm.view(norm.shape[0], 1)
        temp = eps * torch.ones_like(norm)

        x2 = x2 / torch.max(norm, temp)

        # Boradcasting the anchors vector to compute loss over all negative matches
        x1 = x1.unsqueeze(1)
        cos_sim = torch.sum(x1 * x2, dim=2)
        cos_sim = cos_sim / tau

        loss = torch.sum(torch.exp(cos_sim), dim=1)

        return loss

    else:
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        return 1.0 - cos(x1, x2)



# -------------------------------------------------------------------
def plot_avg_acc(test_steps, avg_acc_umaDANN, avg_acc_umaMMD, avg_acc_scratchDANN, avg_acc_scratchMMD, save_file):
    plt.plot(range(test_steps), avg_acc_umaDANN, c='tab:blue', label="UMA-DANN")
    plt.plot(range(test_steps), avg_acc_umaMMD, c='tab:green', label="UMA-MMD")
    plt.plot(range(test_steps), avg_acc_scratchDANN, c='tab:orange', label="Scratch-DANN")
    plt.plot(range(test_steps), avg_acc_scratchMMD, c='tab:purple', label="Scratch-MMD")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Average accuracy", fontsize=12)
    if save_file:
        plt.savefig(save_file)
    plt.close()
    return

# -------------------------------------------------------------------
def plot_worst_acc(test_steps, worst_acc_umaDANN, worst_acc_umaMMD, worst_acc_scratchDANN, worst_acc_scratchMMD, save_file):
    plt.plot(range(test_steps), worst_acc_umaDANN, c='tab:blue', label="UMA-DANN")
    plt.plot(range(test_steps), worst_acc_umaMMD, c='tab:green', label="UMA-MMD")
    plt.plot(range(test_steps), worst_acc_scratchDANN, c='tab:orange', label="Scratch-DANN")
    plt.plot(range(test_steps), worst_acc_scratchMMD, c='tab:purple', label="Scratch-MMD")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Worst accuracy", fontsize=12)
    if save_file:
        plt.savefig(save_file)
    plt.close()
    return