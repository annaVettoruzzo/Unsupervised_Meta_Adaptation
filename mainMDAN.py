import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

import torch, random, numpy as np, os
os.system('nvidia-smi -q -d Memory |grep -A6 GPU|grep Free >tmp')
memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
gpu_number =  int(np.argmax(memory_available))
torch.cuda.set_device(gpu_number)
print(f"Cuda:{gpu_number}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

#################DATASET#####################

dataset_type = "CIFAR10C" #RotatedMNIST, FEMNIST, CIFAR10C, DailySports, WISDM

import sys, os
sys.path.insert(1, "./" + dataset_type)

import data, params, models
from trainingMDAN import evaluate_mdan
from utils import write_in_file

# For reproducibility

torch.random.manual_seed(params.seed)
random.seed(params.seed)
np.random.seed(params.seed)

final_mean_acc = []

for it in range(3):
    PATH = "./" + dataset_type + "/Results" + str(it+1) + "/"
    if not os.path.exists(PATH): os.makedirs(PATH)

    print(f"DATASET {dataset_type}")
    if dataset_type == "RotatedMNIST":
        domains_train_idxs = random.sample(range(params.nb_domains_tot), params.nb_domains_train)
        domains_test_idxs = list(set(range(params.nb_domains_tot)) - set(domains_train_idxs))
        #domains_train_idxs = pd.read_pickle(PATH+"domains_train")
        #domains_test_idxs = pd.read_pickle(PATH+"domains_test")
        rotatedMnist = data.RotatedMNIST(print_output=False)
        domains_train, domains_test = data.domain_generator(rotatedMnist, domains_train_idxs, domains_test_idxs)
        dataset_train = data.RotatedMNIST(domains_idxs = domains_train_idxs, split='train', print_output=False)
        dataset_test = data.RotatedMNIST(domains_idxs = domains_test_idxs, split='test', print_output=False)
        #write_in_file(domains_train_idxs, PATH+"domains_train")
        #write_in_file(domains_test_idxs, PATH+"domains_test")
    elif dataset_type == "FEMNIST":
        dataset_train = data.FEMNISTDataset(split="train")
        domains_train = data.domain_generator(dataset_train, split="train")
        dataset_test = data.FEMNISTDataset(split="test")
        domains_test = data.domain_generator(dataset_test)
    elif dataset_type == "CIFAR10C":
        dataset_train = data.CIFARDataset(split='train')
        domains_train = data.domain_generator(dataset_train)
        dataset_test = data.CIFARDataset(split='test')
        domains_test = data.domain_generator(dataset_test, test='True')
    elif dataset_type == "DailySports":
        root_dir = "DailySports/"
        domains_train_idxs = random.sample(range(params.nb_domains_tot), params.nb_domains_train)
        domains_test_idxs = list(set(range(params.nb_domains_tot)) - set(domains_train_idxs))
        #domains_train_idxs = pd.read_pickle(PATH+"domains_train")
        #domains_test_idxs = pd.read_pickle(PATH+"domains_test")
        HAR_Dataset = data.HARDataset(root_dir, feature_reduction=params.feature_reduction, extract_features=params.extract_features)
        domains_train, domains_test = data.domain_generator(HAR_Dataset, domains_train_idxs, domains_test_idxs)
        dataset_train = data.HARDataset(root_dir, domains_train_idxs, feature_reduction=params.feature_reduction, extract_features=params.extract_features)
        dataset_test = data.HARDataset(root_dir, domains_test_idxs, feature_reduction=params.feature_reduction, extract_features=params.extract_features)
    elif dataset_type == "WISDM":
        root_dir = "WISDM/"
        domains_total = list(range(params.nb_domains_tot))
        #Remove subjects with corrupted data
        domains_total.remove(14)
        domains_total.remove(18)
        domains_total.remove(42)
        domains_train_idxs = random.sample(domains_total, params.nb_domains_train)
        domains_test_idxs = list(set(domains_total)-set(domains_train_idxs))
        WISDM_Dataset = data.WISDMDataset(root_dir, domains_total, datatype = params.data_type, extract_features=params.extract_features)
        domains_train, domains_test = data.domain_generator(WISDM_Dataset, domains_train_idxs, domains_test_idxs)
        dataset_train = data.WISDMDataset(root_dir, domains_train_idxs, datatype=params.data_type, extract_features=params.extract_features)
        dataset_test = data.WISDMDataset(root_dir, domains_test_idxs, datatype=params.data_type, extract_features=params.extract_features)

    ############### MODEL ########################
    mdan_model = models.MDANModel(in_dim=params.input_dim, n_domains=params.nb_domains_train, n_classes=params.n_classes).to(DEVICE)

    if dataset_type=="CIFAR10C" or dataset_type=="FEMNIST":
        n_domains = min(len(domains_train), 8)
    else:
        n_domains = len(domains_train)

    ############### EVALUATE ########################
    test_size = 145 #set the number of labeled samples in the target domain
    test_accuracies, final_accuracy = evaluate_mdan(mdan_model, domains_train, domains_test, n_domains, params.loss, 1., steps=params.test_steps, test_size=test_size)
    print(f"Mean accuracy: {np.mean(final_accuracy)}")
    print(f"Std accuracy: {np.std(final_accuracy)}")
    write_in_file(test_accuracies, PATH+"perf_MDAN")

    final_mean_acc.append(np.mean(final_accuracy))

print("FINAL RESULTS")
print(f"Mean accuracy: {np.mean(final_mean_acc)}")
print(f"Std accuracy: {np.std(final_mean_acc)}")
