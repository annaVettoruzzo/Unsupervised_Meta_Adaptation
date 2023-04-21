from torch.utils.data import Dataset
from collections import defaultdict
from pathlib import Path
import json, os
import torch, random, numpy as np, math
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data.sampler import Sampler

"""
    Download data from:
    - https://drive.google.com/file/d/1xvT13Sl3vJIsC2I7l7Mp8alHkqKQIXaa/view
"""

class FEMNISTDataset(Dataset):
    def __init__(self, split):
        super(FEMNISTDataset, self).__init__()
   
        self.root_dir = Path("./FEMNIST/femnist-data") / split
        clients, _, data = read_dir(self.root_dir) #clients = users
        self.n_domains = len(clients)
        self.domains_idxs = list(range(self.n_domains))
        
        self.num_classes = 62
        
        self.image_shape = (1, 28, 28)
        
        agg_X, agg_y, agg_domains = [], [], []
        for i, client in enumerate(clients):
            client_X, client_y = data[client]['x'], data[client]['y']
            client_N = len(client_X)
            X_processed = np.array(client_X).reshape((client_N, 28, 28, 1))
            X_processed = (1.0 - X_processed)
            agg_X.append(X_processed)
            agg_y.extend(client_y)
            agg_domains.extend([i] * client_N)
        self._len = len(agg_domains)
        self._X, self._y = np.concatenate(agg_X), np.array(agg_y)
        self.domain_ids = np.array(agg_domains)
        
        self.transform = get_transform()
        
        self.domain_counts, _ = np.histogram(self.domain_ids, bins=range(self.n_domains + 1), density=False)
        
        # Print dataset stats       
        print("Number of examples", self._len)
        print("Number of classes", self.num_classes)  
        print("Number of domains", self.n_domains)  
        
        print("Smallest domain: ", np.min(self.domain_counts))
        print("Largest domain: ", np.max(self.domain_counts))
           
        
    def __len__(self):
        """Returns number of examples in the dataset"""
        return self._len
    
    def __getitem__(self, index):
        
        img = self.transform(**{'image': self._X[index]})['image'] #color channel is the first dimension
        img = img.type(torch.float)
        
        label = torch.tensor(self._y[index], dtype=torch.long)
        
        domain_id = torch.tensor(self.domain_ids[index], dtype=torch.long)

        return img, label, domain_id
    
# -------------------------------------------------------------------
def domain_generator(dataset, split="test", k=0.3, q=0.7):
    domains = []
    
    for i in range(dataset.n_domains):
        X, y = [], []
        idxs = np.where(dataset.domain_ids == i)[0].tolist()      

        for j in range(len(idxs)):
            X.append(dataset[idxs[j]][0])
            y.append(dataset[idxs[j]][1])

        # Split the data into support set (k examples per class) and query set (q examples per class)   
        sp_idx = random.sample(range(len(X)), int(k * len(X)))
        qr_idx = random.sample(list(set(list(range(len(X)))) - set(sp_idx)), int(q * len(X)))

        X_sp = [X[v] for v in sp_idx]
        y_sp = [y[v] for v in sp_idx]       
        X_qr = [X[v] for v in qr_idx]
        y_qr = [y[v] for v in qr_idx]


        if split=="train": domains.append((torch.stack(X_sp), torch.tensor(y_sp), torch.stack(X_qr), torch.tensor(y_qr), i))          
        else: domains.append((torch.stack(X_sp), torch.tensor(y_sp), i))

    return domains

# -------------------------------------------------------------------
def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data

# -------------------------------------------------------------------
def get_transform():
    transform = albumentations.Compose([ToTensorV2()])
    return transform
 
#before shuffling keep only the samples with the same label in both domains
def shuffle_labels(domain_s, domain_t, k=0.3, q=0.7):     
    (Xs_sp, ys_sp, Xs_qr, ys_qr, ds), (Xt_sp, yt_sp, Xt_qr, yt_qr, dt) = domain_s, domain_t
    Xs, ys = torch.cat((Xs_sp, Xs_qr)), torch.cat((ys_sp, ys_qr))
    Xt, yt = torch.cat((Xt_sp, Xt_qr)), torch.cat((yt_sp, yt_qr))

    #remove classes not in common and make the two domains with the same length
    Xs, ys, Xt, yt = process_domains(Xs, ys, Xt, yt)
    common_labels = list(set(np.unique(ys)).intersection(np.unique(yt)))
            
    #shuffle the labels
    new_ys = torch.tensor([0] * len(ys))
    new_yt = torch.tensor([0] * len(yt))
    new_labels = random.sample(common_labels,len(common_labels))
    for i in range(len(common_labels)):
        for v in np.where(ys == common_labels[i])[0]: new_ys[v] = new_labels[i] 
        for v in np.where(yt == common_labels[i])[0]: new_yt[v] = new_labels[i] 

    #divide into support and query. Each set must have the same labels
    sp_idx = random.sample(range(len(Xs)), int(k * len(Xs)))
    qr_idx = random.sample(list(set(list(range(len(Xs)))) - set(sp_idx)), math.ceil(q * len(Xs)))
    
    Xs_sp, ys_sp, Xs_qr, ys_qr = [], [], [], []
    Xt_sp, yt_sp, Xt_qr, yt_qr = [], [], [], []
    for i in sp_idx:
        Xs_sp.append(Xs[i])
        ys_sp.append(new_ys[i])
        idxs = np.where(new_yt == new_ys[i])[0]
        if len(idxs) > 1: idx = random.choice(idxs)
        else: idx = idxs[0]
        Xt_sp.append(Xt[idx])
        yt_sp.append(new_yt[idx])

    for i in qr_idx:
        Xs_qr.append(Xs[i])
        ys_qr.append(new_ys[i])
        idxs = np.where(new_yt == new_ys[i])[0]
        if len(idxs) > 1: idx = random.choice(idxs)
        else: idx = idxs[0]
        Xt_qr.append(Xt[idx])
        yt_qr.append(new_yt[idx])
    
    #Shuffle one of the two domains
    c = list(zip(Xs_sp, ys_sp))
    random.shuffle(c)
    Xs_sp, ys_sp = zip(*c)
    
    c = list(zip(Xs_qr, ys_qr))
    random.shuffle(c)
    Xs_qr, ys_qr = zip(*c)
    
    return (torch.stack(Xs_sp), torch.tensor(ys_sp), torch.stack(Xs_qr), torch.tensor(ys_qr), ds), (torch.stack(Xt_sp), torch.tensor(yt_sp), torch.stack(Xt_qr), torch.tensor(yt_qr), dt)

def process_domains(Xs, ys, Xt, yt):
    #remove elements not in both domains
    label_s, label_t = np.unique(ys), np.unique(yt)
    common_labels = list(set(label_s).intersection(label_t))
    if not common_labels: #if the list is Empty
        raise Exception("No common values between S and T")
    uncommon_labels = list(set(range(62))-set(common_labels))
    for i in range(len(uncommon_labels)):
        value = uncommon_labels[i]        
        if value in ys:
            Xs, ys = Xs[ys!=value], ys[ys!=value] #remove the element from Xs
        if value in yt:
            Xt, yt = Xt[yt!=value], yt[yt!=value] #remove the element from Xt
            
    #make the two domains with the same length
    if len(Xs) < len(Xt):
        m = len(Xt)-len(Xs)
        for i in range(m):
            r = random.choice(common_labels) 
            while len(Xt[yt==r]) <= 1 or len(Xt[yt==r]) == len(Xs[ys==r]): r = random.choice(common_labels)
            idx = np.where(yt==r)[0][0]
            Xt = torch.cat([Xt[:idx], Xt[idx+1:]])
            yt = torch.cat([yt[:idx], yt[idx+1:]])
    elif len(Xs) > len(Xt):
        m = len(Xs)-len(Xt)
        for i in range(m):
            r = random.choice(common_labels) 
            while len(Xs[ys==r]) <= 1 or len(Xs[ys==r]) == len(Xt[yt==r]): r = random.choice(common_labels)
            idx = np.where(ys==r)[0][0]
            Xs = torch.cat([Xs[:idx], Xs[idx+1:]])
            ys = torch.cat([ys[:idx], ys[idx+1:]])
    return Xs, ys, Xt, yt        
            
            
# -------------------------------------------------------------------
""" Functions for training ARM-CML."""
# -------------------------------------------------------------------

class GroupSampler:
    """
        Samples batches of data from predefined groups.
    """

    def __init__(self, dataset, meta_batch_size, support_size,
                 drop_last=None, uniform_over_groups=True):

        self.dataset = dataset
        self.indices = range(len(dataset))

        self.domain_ids = dataset.domain_ids
        self.domains = dataset.domains_idxs
        self.num_domains = dataset.n_domains

        self.meta_batch_size = meta_batch_size
        self.support_size = support_size
        self.batch_size = meta_batch_size * support_size
        self.drop_last = drop_last
        self.dataset_size = len(self.dataset)
        self.num_batches = len(self.dataset) // self.batch_size

        self.domains_with_ids = {}
        self.actual_domains = []

        # group_count will have one entry per group
        # with the size of the group
        self.domain_count = []
        for group_id in self.domains:
            ids = np.nonzero(self.domain_ids == group_id)[0]
            self.domain_count.append(len(ids))
            self.domains_with_ids[group_id] = ids

        self.domain_count = np.array(self.domain_count)
        self.domain_prob = self.domain_count / np.sum(self.domain_count)
        self.uniform_over_groups = uniform_over_groups

    def __iter__(self):

        n_batches = len(self.dataset) // self.batch_size
        if self.uniform_over_groups:
            sampled_domains = np.random.choice(self.domains, size=(n_batches, self.meta_batch_size))
        else:
            # Sample groups according to the size of the group
            sampled_domains = np.random.choice(self.domains, size=(n_batches, self.meta_batch_size), p=self.domain_prob)

        domain_sizes = np.zeros(sampled_domains.shape)

        for batch_id in range(self.num_batches):

            sampled_ids = [np.random.choice(self.domains_with_ids[sampled_domains[batch_id, sub_batch]],
                                size=self.support_size,
                                replace=True,
                                p=None)
                                for sub_batch in range(self.meta_batch_size)]



            # Flatten
            sampled_ids = np.concatenate(sampled_ids)

            yield sampled_ids

        self.sub_distributions = None

    def __len__(self):
        return len(self.dataset) // self.batch_size

# -------------------------------------------------------------------
def get_loader(dataset, sampler_type, uniform_over_groups=False,
               meta_batch_size=None,support_size=None, shuffle=True,
               pin_memory=True, num_workers=8, args=None):
    
    if sampler_type == 'group': # Sample support batches from multiple sub distributions
        batch_sampler = GroupSampler(dataset, meta_batch_size, support_size,
                                      uniform_over_groups=uniform_over_groups)

        batch_size = 1
        shuffle = None
        sampler=None
        drop_last = False
    else:

        batch_size = meta_batch_size * support_size

        if uniform_over_groups:
            group_weights = 1 / dataset.domain_counts
            weights = group_weights[dataset.domain_ids]
            sampler = torch.utils.data.WeightedRandomSampler(weights, len(dataset), replacement=True)
            batch_sampler = None
            drop_last = True
            shuffle = None
        else: # Sample each example uniformly

            print("standard sampler")

            sampler = None
            batch_sampler = None
            if args is not None:
                drop_last = bool(args.drop_last)
            else:
                drop_last = False
            if shuffle == 0:
                shuffle=False
            else:
                shuffle=True
            print("shuffle: ", shuffle)
    
    loader = torch.utils.data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  sampler=sampler,
                                  batch_sampler=batch_sampler,
                                  pin_memory=pin_memory,
                                  num_workers=num_workers,
                                  drop_last=drop_last)
    return loader