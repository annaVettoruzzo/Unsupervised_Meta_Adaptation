import torch, math, numpy as np, random
from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms as transforms

"""
    Download data from:
    - https://drive.google.com/file/d/1blM7LHGR62-dVJjNAScsJMlzeiQS9DX1/view?usp=sharing for training
    - https://zenodo.org/record/2535967#.YCUsMukzZ0s for testing
"""

def load_corruption(path):
    data = np.load(path)
    return np.array(np.array_split(data, 5))

class CIFARDataset(Dataset):
    def __init__(self, split): 
        super(CIFARDataset, self).__init__()
        
        if split == 'train':
            self.root_dir = Path('./CIFAR10C/CIFAR-10-C-new/train/')
            corruptions = ['gaussian_noise', 'shot_noise', 'defocus_blur', 'glass_blur', 'zoom_blur', 'snow', 'frost', 'brightness', 'contrast', 'pixelate']
            other_idx = [0, 1, 2, 5, 6, 7]
        if split == 'test':
            self.root_dir = Path("./CIFAR10C/CIFAR-10-C/") 
            corruptions = ['impulse_noise', 'motion_blur', 'fog', 'elastic_transform']
            other_idx = [4, 8]
            
        other = [load_corruption(self.root_dir / (corruption + '.npy')) for corruption in ['spatter', 'jpeg_compression']]
        other = np.concatenate(other, axis=0)[other_idx]

        data = [load_corruption(self.root_dir / (corruption + '.npy')) for corruption in corruptions]
        data = np.concatenate(data, axis=0)
        
        self._X = np.concatenate([other, data], axis=0)

        n_images_per_domain = self._X.shape[1]
        
        self.n_domains = self._X.shape[0]
        self.domains = list(range(self.n_domains))
        self.domains_idxs = self.domains
        self.image_shape = (3, 32, 32)
        self._X = self._X.reshape((-1, 32, 32, 3))
        self.num_classes = 10
        
        if split == 'test':
            n_images = 10000      
            self._y = np.load(self.root_dir / 'labels.npy')[:n_images]
            self._y = np.tile(self._y, self.n_domains)
            self.domain_ids = np.array([[i]*n_images for i in range(self.n_domains)]).flatten()
        else:
            n_images = 1000
            other_labels = [load_corruption(self.root_dir / (corruption + '_labels.npy')) for corruption in ['spatter', 'jpeg_compression']]
            other_labels = np.concatenate(other_labels, axis=0)[other_idx]
            data_labels = [load_corruption(self.root_dir / (corruption + '_labels.npy')) for corruption in corruptions]
            data_labels = np.concatenate(data_labels, axis=0)
            self._y = np.concatenate([other_labels, data_labels], axis=0).flatten()
            self.domain_ids = np.array([[i]*n_images for i in range(self.n_domains)]).flatten()
            
        self._len = len(self.domain_ids)
        
        self.domain_counts, _ = np.histogram(self.domain_ids,
                                            bins=range(self.n_domains + 1),
                                            density=False)
        self.transform = get_transform()      
        print("Dataset size: ", len(self._y))
        print("Number of classes: ", self.num_classes)
        print("Number of domains: ", self.n_domains)
        print("Smallest group: ", np.min(self.domain_counts))
        print("Largest group: ", np.max(self.domain_counts))
        
    def __len__(self):
        """Returns number of examples in the dataset"""
        return self._len
    
    def __getitem__(self, index):
        
        #img = self.transform(**{'image': self._X[index]})['image'] #color channel is the first dimension
        img = self.transform(self._X[index])
        img = img.type(torch.float)
        
        label = torch.tensor(self._y[index], dtype=torch.long)
        
        domain_id = torch.tensor(self.domain_ids[index], dtype=torch.long)

        return img, label, domain_id

def get_transform():
    transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return transform

def domain_generator(dataset, k = 0.3, q = 0.7, test='False'): 
    domains = []
    
    for i in range(dataset.n_domains):
        X, y = [], []      
        if test: idxs = random.sample(np.where(dataset.domain_ids == i)[0].tolist(), 1000)
        else: idxs = np.where(dataset.domain_ids == i)[0].tolist()         
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

        if test == 'False':
            domains.append((torch.stack(X_sp), torch.tensor(y_sp), torch.stack(X_qr), torch.tensor(y_qr), i))          
        else:    
            domains.append((torch.stack(X_sp), torch.tensor(y_sp), i))
            
    return domains

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
    def __init__(self, dataset, meta_batch_size, support_size,
                 drop_last=None, uniform_over_groups=True):

        self.dataset = dataset
        self.indices = range(len(dataset))

        self.domain_ids = dataset.domain_ids
        self.domains = dataset.domains
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