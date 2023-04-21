import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

import torch, random, numpy as np, os, pandas as pd
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

#################DATASET#####################

dataset_type = "WISDM" #RotatedMNIST, FEMNIST, CIFAR10C, DailySports, WISDM

import sys, os
sys.path.insert(1, "./" + dataset_type)

import params

#Plot
models_list = ["perf_umaDANN", "perf_umaMMD", "perf_MDAN", "perf_scratchDANN", "perf_scratchMMD"]
average_accuracy = []
std_error = []
for perf in models_list:
    avg_acc = []
    for i in range(3):
        file_name = "./"+dataset_type+"/Results"+str(i+1)+"/"+perf
        if perf == "perf_MDAN":
            test_accuracies = pd.read_pickle(file_name)
            avg_acc.append(np.mean(test_accuracies, axis=0))
        else:
            avg_acc.append(list(pd.read_pickle(file_name)['average accuracy']))
    average_accuracy.append(np.mean(np.array(avg_acc),0))
    std_error.append(np.std(np.array(avg_acc),0))


plt.plot(range(params.test_steps), average_accuracy[0], c='tab:blue', label="UMA-DANN")
plt.fill_between(range(params.test_steps), average_accuracy[0]-std_error[0], average_accuracy[0]+std_error[0], color='tab:blue', alpha=0.2)
plt.plot(range(params.test_steps), average_accuracy[1], c='tab:green', label="UMA-MMD")
plt.fill_between(range(params.test_steps), average_accuracy[1]-std_error[1], average_accuracy[1]+std_error[1], color='tab:green', alpha=0.2)
plt.plot(range(params.test_steps), average_accuracy[2], c='tab:pink', label="MDAN")
plt.fill_between(range(params.test_steps), average_accuracy[2]-std_error[2], average_accuracy[2]+std_error[2], color='tab:pink', alpha=0.2)
plt.plot(range(params.test_steps), average_accuracy[3], c='tab:orange', label="DANN")
plt.fill_between(range(params.test_steps), average_accuracy[3]-std_error[3], average_accuracy[3]+std_error[3], color='tab:orange', alpha=0.2)
plt.plot(range(params.test_steps), average_accuracy[4], c='tab:purple', label="MMD")
plt.fill_between(range(params.test_steps), average_accuracy[4]-std_error[4], average_accuracy[4]+std_error[4], color='tab:purple', alpha=0.2)
plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.legend(loc=4)
plt.title("Average accuracy", fontsize=12)
plt.show()