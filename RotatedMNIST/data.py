import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random, numpy as np, pandas as pd
from tabulate import tabulate
from  scipy import ndimage

N_DOMAINS = 14 

config = {}
config['group_type'] = 'rotation'
config['n_domains'] = N_DOMAINS
config['domain_values'] = np.array(range(N_DOMAINS)) * 10
config['domain_probs'] = np.zeros(N_DOMAINS)
config['domain_probs'][:3] = 7/100 #0 - 20
config['domain_probs'][3:6] = 7/100 #30 - 50
config['domain_probs'][6:9] = 7/100 #60 - 80
config['domain_probs'][9:12] = 7/100 #90 - 110
config['domain_probs'][12:] = 7/100 #120 - 130

# -------------------------------------------------------------------
class RotatedMNIST(Dataset):
    def __init__(self, domains_idxs=None, split=None, print_output=True): 
        super(RotatedMNIST, self).__init__()
        
        if split == 'train': #for ARM-CML
            data = datasets.MNIST(root='RMNIST', train=True, download=True)
        elif split == 'test': #for ARM-CML
            data = datasets.MNIST(root='RMNIST', train=False, download=True)
        else: 
            data = datasets.MNIST(root='RMNIST', train=True, download=True)
  
        self.images = data.data
        self.labels = data.targets
        self.original_size = len(self.images)
        self.num_classes = 10
        self.all_indices = range(self.original_size)
        if domains_idxs is not None: 
            self.domains_idxs = domains_idxs 
            
        self.n_domains = config['n_domains']
        self.domain_values = config['domain_values']   
        self.config = config      
        self.domains = list(range(self.n_domains))
        
        if split == 'train' or split == 'test': #for ARM-CML
            self.indices, self.domain_ids = self.get_domains(config)
        else: 
            self.indices, self.domain_ids = self.get_all_domains(config)
    
        # Retrieve set
        self.images, self.labels = self.images[self.indices], self.labels[self.indices]
        
        # Map to domains
        self.domain_stats = np.zeros((config['n_domains'], 2))
        
        for i in range(config['n_domains']):
            idxs = np.nonzero(np.asarray(self.domain_ids == i))[0]
            num_in_domain = len(idxs)
            self.domain_stats[i, 0] = num_in_domain
            self.domain_stats[i, 1] = num_in_domain / len(self.labels) # Fraction in domain
         
        self.df_stats = pd.DataFrame(self.domain_stats, columns=['n', 'frac'])
        self.df_stats['domain_id'] = self.df_stats.index
        self.df_stats['degree'] = self.domain_values[self.df_stats.index]
           
        # Print dataset stats
        if print_output:
            print("Number of examples", len(self.indices))
            print(tabulate(self.df_stats, headers='keys', tablefmt='psql'))           
    
    def get_domains(self, config):
        """Returns only some domains"""
        indices = []
        domain_ids = []
        for d in range(config['n_domains']):        
            if d in self.domains_idxs: domain_prob = config['domain_probs'][d]
            else: continue
            num_examples = int(domain_prob * self.original_size / 5)
            indices_for_domain = np.random.choice(self.original_size, size=num_examples)
            domain_ids.append(len(indices_for_domain) * [d])
            indices.append(indices_for_domain)  
            
        domain_ids = np.concatenate(domain_ids)
        indices = np.concatenate(indices) 
        
        return indices, domain_ids
    
    def get_all_domains(self, config):
        """Returns all domains"""
        indices = []
        domain_ids = []
        for d in range(config['n_domains']):
            domain_prob = config['domain_probs'][d]
            if domain_prob == 0:
                continue
            num_examples = int(domain_prob * self.original_size/5)
            indices_for_domain = np.random.choice(self.original_size, size=num_examples)
            domain_ids.append(len(indices_for_domain) * [d])
            indices.append(indices_for_domain)   
            
        domain_ids = np.concatenate(domain_ids)
        indices = np.concatenate(indices)
        
        return indices, domain_ids
                                                                    
    def __len__(self):
        """Returns number of examples in the dataset"""
        return len(self.labels)   
    
    def __getitem__(self, index):
        domain_id = self.domain_ids[index]
        img = self.images[index]
        self.apply_transform = True
        if self.apply_transform:            
            # Rotate
            domain_value = self.domain_values[domain_id]
            img = rotate(img, domain_value, single_image=True)    
            
        # Normalize
        img = rescale(img) 
        
        # Convert to tensor
        img = torch.tensor(img, dtype=torch.float)
        
        # Put color channel first
        img = torch.unsqueeze(img, dim=0)
        
        label = self.labels[index]
        
        return img, label, domain_id

# -------------------------------------------------------------------
''' Apply a rotation to a batch of images or a single image. '''
def rotate(X, rotation, single_image = False):
    if single_image:
        return np.array(ndimage.rotate(X, rotation, reshape=False, order=0))
    else:
        return np.array([ndimage.rotate(X[i], rotation[i], reshape=False, order=0) for i in range(X.shape[0])])    
    
def rescale(X):
    return X.astype(np.float32) / 255.

# -------------------------------------------------------------------
"""
    Select domains for training or testing. 
    k = percentage of examples in the support set. q = percentage of examples in the test set.
"""
def domain_generator(dataset, domains_train_idxs, domains_test_idxs, k = 0.3, q = 0.7):
    domains = []
    domains_test = []

    # create each task = group = domain   
    for i in range(N_DOMAINS):
        X, y = [], []
        idxs = np.where(dataset.domain_ids == i)[0]
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

        if i in domains_train_idxs:
            domains.append((torch.stack(X_sp), torch.tensor(y_sp), torch.stack(X_qr), torch.tensor(y_qr), i))          
        else:    
            domains_test.append((torch.stack(X_sp), torch.tensor(y_sp), i))

    return domains, domains_test

# -------------------------------------------------------------------
"""
    Shuffle domain labels to avoid the memorization problem. 
    Each pair of S and T must be shuffled consistently.
"""
def shuffle_labels(domain_s, domain_t):
    (Xs_sp, ys_sp, Xs_qr, ys_qr, ds), (Xt_sp, yt_sp, Xt_qr, yt_qr, dt) = domain_s, domain_t
    Xs, ys = torch.cat((Xs_sp, Xs_qr)), torch.cat((ys_sp, ys_qr))
    Xt, yt = torch.cat((Xt_sp, Xt_qr)), torch.cat((yt_sp, yt_qr))

    new_ys = torch.tensor([0] * len(ys))
    new_yt = torch.tensor([0] * len(yt))
    new_labels = list(range(10))
    random.shuffle(new_labels)
    for i in list(range(10)):
        new_ys[np.where(ys == i)[0]] = new_labels[i]
        new_yt[np.where(yt == i)[0]] = new_labels[i]
        
    return (Xs[:len(Xs_sp)], new_ys[:len(ys_sp)], Xs[len(Xs_sp):], new_ys[len(ys_sp):], ds), (Xt[:len(Xt_sp)], new_yt[:len(yt_sp)], Xt[len(Xt_sp):], new_yt[len(yt_sp):], dt)


# -------------------------------------------------------------------
""" Functions for training ARM-CML."""
# -------------------------------------------------------------------
class GroupSampler:
    """ Samples batches of data from predefined groups. """

    def __init__(self, dataset, meta_batch_size, support_size, drop_last=None, uniform_over_groups=True):

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

        # group_count will have one entry per group with the size of the group
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


