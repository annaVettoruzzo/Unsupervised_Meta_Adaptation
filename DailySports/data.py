import os
import pandas as pd, numpy as np, random, math
from utils import write_in_file, convert_into_int
from torch.utils.data import Dataset
import torch
from sklearn.decomposition import PCA

"""
    Download data from:
    -  https://archive.ics.uci.edu/ml/datasets/daily+and+sports+activities
"""

patients = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
activities = ['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19']

# -------------------------------------------------------------------
# Domains are patients; Classes are different activities  
class HARDataset(Dataset):
    def __init__(self, root_dir, domains_idxs=None, feature_reduction = True, extract_features=False): 
        super(HARDataset, self).__init__()  
        
        self.root_dir = root_dir
        self.patients = patients
        self.activities = activities
        if extract_features: 
            self.normalized_values_list, self.patient_label_list, self.activity_label_list = self.prepare_dataset()
        else: 
            self.normalized_values_list = pd.read_pickle(self.root_dir+"normalized_features")
            self.patient_label_list = pd.read_pickle(self.root_dir+"patients")
            self.activity_label_list = pd.read_pickle(self.root_dir+"activities")    
        self.n_classes = len(self.activity_label_list)
        self.all_indices = range(len(self.normalized_values_list))     
        
        if domains_idxs == None: #create a dataset with all the possible tasks
            self.n_domains = len(np.unique(self.patient_label_list)) 
            self.domains_idxs = list(range(self.n_domains))
        else: 
            self.n_domains = len(domains_idxs) # the dataset will consider only specific tasks
            self.domains_idxs = domains_idxs
        
        self.indices, self.domain_ids = self.get_domains()
        
        # Retrieve set
        self.features, self.activity_label_list = np.array(self.normalized_values_list)[self.indices], np.array(self.activity_label_list)[self.indices]
        self.labels = convert_into_int(self.activity_label_list)
        
        # Apply PCA
        if feature_reduction:
            print("PCA reduction")
            pca = PCA(n_components=30)
            pca.fit(self.features)
            self.features = pca.transform(self.features)
            
    def prepare_dataset(self):
        normalized_values_list = []
        patient_label_list = []
        activity_label_list = []

        for activity in self.activities:    
            os.chdir(self.root_dir+"/data")
            # going inside each activity folder
            os.chdir(activity)
            patient_files = os.listdir()
            for patient in patient_files:      
                # going into every patient folder
                os.chdir(patient)
                segment_files = os.listdir()
                for filename in segment_files:         
                    # obtaining the 1170x1 vector, patient id, activity id from the text file.
                    features = feature_extraction(filename)
                    normalized_features = normalize(features)

                    normalized_values_list.append(list(normalized_features)) # a 2D list with 9120 lists insdie it, each has 1170 values.
                    patient_label_list.append(patient) # a 1D list with 9120 patient ids.
                    activity_label_list.append(activity) # a 1D list iwth 9120 activity ids.

                os.chdir(self.root_dir+"/data/" + activity)

        os.chdir(self.root_dir)
        
        write_in_file(normalized_values_list, "normalized_features")
        write_in_file(patient_label_list, "patients")
        write_in_file(activity_label_list, "activities")

        return normalized_values_list, activity_label_list, patient_label_list

    def get_domains(self):
        indices = []
        domain_ids = []
        for idx in self.domains_idxs:
            indices_for_domain = np.where(np.array(self.patient_label_list) == self.patients[idx])[0] 
            domain_ids.append(len(indices_for_domain) * [idx])
            indices.append(indices_for_domain)   
            
        domain_ids = np.concatenate(domain_ids)
        indices = np.concatenate(indices)
        return indices, domain_ids
    
    def __len__(self):
        """Returns number of examples in the dataset"""
        return len(self.labels)   
    
    def __getitem__(self, index):
        domain_id = self.domain_ids[index]
        feature = self.features[index]
        label = self.labels[index]
        return feature, label, domain_id
    
# -------------------------------------------------------------------    
def feature_extraction(filename):
    
    data = pd.read_csv('{}'.format(filename), header=None)
    
    #225 statistical features
    first_step = list(data.min())+list(data.max())+list(data.mean())+list(data.skew())+list(data.kurtosis())
    
    # FFT features
    dft_matrix = np.fft.fft(data.to_numpy().T)
    abs_dft_matrix = np.absolute(dft_matrix)
    f_k = (2*math.pi)/5
    
    second_step =[]
    third_step = []
    for i in range(len(abs_dft_matrix)):
        positions = abs_dft_matrix[i].argsort()[-5:][::-1]
        second_step.append(list(abs_dft_matrix[i][positions]))
        third_step.append(list(positions*f_k))
    second_step = [item for sublist in second_step for item in sublist] # flattening the lists.
    third_step = [item for sublist in third_step for item in sublist]

    # Autocorrelation
    fourth_step = []
    autocorr_reqd = [0,4,9,14,19,24,29,34,39,44,49] # getting the required autocorr values as mentioned in the paper.
    for column in data.columns:
        mean = data[column].mean()
        for delta in range(len(data)):
            if(delta in autocorr_reqd):
                sum_of_products = 0
                for i, row in enumerate(data[column], start = delta):
                    element_1 = row - mean
                    element_2 = data[column].iloc[len(data)-1-i] - mean
                    sum_of_products += element_1*element_2
                rss = 1/(len(data)-delta)*sum_of_products 
                fourth_step.append(rss)

    # Make result
    final_representation = first_step + second_step + third_step + fourth_step
    return final_representation

# -------------------------------------------------------------------
def normalize(features):
    arr = np.asarray(features)
    normalized = (arr-min(arr))/(max(arr)-min(arr))
    return normalized

# -------------------------------------------------------------------
'''
    Select domains for training or testing. 
    k = percentage of examples in the support set. q = percentage of examples in the test set.
'''
def domain_generator(dataset, domains_train_idxs, domains_test_idxs, k = 0.3, q = 0.7):
    domains = []
    domains_test = []

    # create each task = group = domain   
    for i in range(len(patients)):
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
            domains.append((torch.FloatTensor(np.array(X_sp)), torch.LongTensor(np.array(y_sp)), torch.FloatTensor(np.array(X_qr)), torch.LongTensor(np.array(y_qr)), i))          
        else:    
            domains_test.append((torch.FloatTensor(np.array(X_sp)), torch.LongTensor(np.array(y_sp)), i))

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
    new_labels = random.sample(list(np.unique(ys)),len(np.unique(ys)))
    for i in range(len(np.unique(ys))):
        new_ys[np.where(ys == i)[0]] = new_labels[i]
        new_yt[np.where(yt == i)[0]] = new_labels[i]
        
    return (Xs[:len(Xs_sp)], new_ys[:len(ys_sp)], Xs[len(Xs_sp):], new_ys[len(ys_sp):], ds), (Xt[:len(Xt_sp)], new_yt[:len(yt_sp)], Xt[len(Xt_sp):], new_yt[len(yt_sp):], dt)


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