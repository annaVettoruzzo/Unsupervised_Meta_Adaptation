import os
import pandas as pd, numpy as np, random
from utils import write_in_file, convert_into_int
from torch.utils.data import Dataset
from pathlib import Path
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder

"""
    Download data from:
    - https://github.com/ma-shamshiri/Human-Activity-Recognition/tree/main/code/data/transformed_data
"""

#Consider only one eating activity (i.e. eat sandwich)
activity_dict= {"A":"Walking",
                "B":"Jogging",
                "C":"Stairs",
                "D":"Sitting",
                "E":"Standing",
                "F":"Typing",
                "G":"Brushing",
                #"H":"Eat Soup",
                #"I":"Eat Chips",
                #"J":"Eat Pasta",
                "K":"Drinking",
                "L":"Eat Sandwich",
                "M":"Kicking",
                "O":"Playing",
                "P":"Dribblinlg",
                "Q":"Writing",
                "R":"Clapping",
                "S":"Folding"}

# Domains are patients; Classes are different activities  
class WISDMDataset(Dataset):
    def __init__(self, root_dir, patients, domains_idxs=None, datatype = 'both', extract_features=False): #datatype can be 'both', 'phone' or 'watch'
        super(WISDMDataset, self).__init__()  
        
        self.root_dir = root_dir
        self.datatype = datatype
        self.data_list = self.store_data()
        self.patients = patients
        self.activities = list(activity_dict.keys())
        if extract_features: 
            self.normalized_values_list, self.activity_label_list, self.patient_label_list = self.prepare_dataset()
        else: 
            self.normalized_values_list = pd.read_pickle(self.root_dir+"normalized_features")
            self.patient_label_list = pd.read_pickle(self.root_dir+"patients")
            self.activity_label_list = pd.read_pickle(self.root_dir+"activities") 
            
        if domains_idxs == None: #create a dataset with all the possible tasks
            self.n_domains = len(self.patients) 
            self.domains_idxs = self.patients
        else: 
            self.n_domains = len(domains_idxs) # the dataset will consider only specific tasks
            self.domains_idxs = domains_idxs
        
        self.indices, self.domain_ids = self.get_domains()
        
        # Retrieve set
        self.features, self.labels = np.array(self.normalized_values_list)[self.indices], np.array(self.activity_label_list)[self.indices]
            
    def store_data(self):
        phone_accel_file_paths = []
        phone_gyro_file_paths = []
        watch_accel_file_paths = []
        watch_gyro_file_paths = []

        for directories, subdirectories, files in os.walk(self.root_dir+"/transformed_data/"):
            for filename in files:
                subj_id = filename.split("_")
                if len(subj_id)>1 and subj_id[1]!='1614' and subj_id[1] !='1618' and subj_id[1]!='1642':
                    if "phone" in filename and "accel" in filename:
                        phone_accel_file_paths.append(f"{self.root_dir}phone/accel/{filename}")
                    elif "phone" in filename and "gyro" in filename:
                        phone_gyro_file_paths.append(f"{self.root_dir}phone/gyro/{filename}")
                    elif "watch" in filename and "accel" in filename:
                        watch_accel_file_paths.append(f"{self.root_dir}watch/accel/{filename}")
                    elif "watch" in filename and "gyro" in filename:
                        watch_gyro_file_paths.append(f"{self.root_dir}watch/gyro/{filename}")
                    
        data_list = []
        if self.datatype == 'accel':
            data_list.extend(phone_accel_file_paths)
            data_list.extend(watch_accel_file_paths)
        elif self.datatype == 'gyro':
            data_list.extend(phone_gyro_file_paths)
            data_list.extend(watch_gyro_file_paths)
        else:
            data_list.extend(phone_accel_file_paths)
            data_list.extend(watch_accel_file_paths)
            data_list.extend(phone_gyro_file_paths)
            data_list.extend(watch_gyro_file_paths)
        return data_list
    
    def prepare_dataset(self):
        normalized_values_list = []
        patient_label_list = []
        activity_label_list = []
        
        for subjectid, file in enumerate(self.data_list[:]):
            subjectid = file.split("_")[2]
            data = pd.read_csv(file, verbose=False)
            normalized_features, activity_labels = self.preprocess_data(data)

            normalized_values_list.append(normalized_features)
            patient_label_list.append(len(activity_labels) * [int(subjectid)-1600])
            activity_label_list.append(activity_labels)

        write_in_file(np.concatenate(normalized_values_list), "normalized_features")
        write_in_file(np.concatenate(patient_label_list), "patients")
        write_in_file(np.concatenate(activity_label_list), "activities")
        
        return np.concatenate(normalized_values_list), np.concatenate(activity_label_list), np.concatenate(patient_label_list)
    
    def balance_data(self, dataframe):
        """  Take only the first 17 data rows for each activity """
        A = dataframe[dataframe['ACTIVITY']=='A'].head(17).copy()
        B = dataframe[dataframe['ACTIVITY']=='B'].head(17).copy()
        C = dataframe[dataframe['ACTIVITY']=='C'].head(17).copy()
        D = dataframe[dataframe['ACTIVITY']=='D'].head(17).copy()
        E = dataframe[dataframe['ACTIVITY']=='E'].head(17).copy()
        F = dataframe[dataframe['ACTIVITY']=='F'].head(17).copy()
        G = dataframe[dataframe['ACTIVITY']=='G'].head(17).copy()
        #H = dataframe[dataframe['ACTIVITY']=='H'].head(17).copy()
        #I = dataframe[dataframe['ACTIVITY']=='I'].head(17).copy()
        #J = dataframe[dataframe['ACTIVITY']=='J'].head(17).copy()
        K = dataframe[dataframe['ACTIVITY']=='K'].head(17).copy()
        L = dataframe[dataframe['ACTIVITY']=='L'].head(17).copy()
        M = dataframe[dataframe['ACTIVITY']=='M'].head(17).copy()
        O = dataframe[dataframe['ACTIVITY']=='O'].head(17).copy()
        P = dataframe[dataframe['ACTIVITY']=='P'].head(17).copy()
        Q = dataframe[dataframe['ACTIVITY']=='Q'].head(17).copy()
        R = dataframe[dataframe['ACTIVITY']=='R'].head(17).copy()
        S = dataframe[dataframe['ACTIVITY']=='S'].head(17).copy()

        balanced_data = pd.DataFrame()
        balanced_data = balanced_data.append([A, B, C, D, E, F, G, K, L, M, O, P, Q, R, S], ignore_index=True)
        return balanced_data

    def clean_data(self, dataframe):
        """ Remove the columns "ACTIVITY" and "class" from the dataframe """
        """ Take only 43 important features of the dataframe  """
        df = dataframe.drop(['ACTIVITY', 'class'], axis = 1).copy()
        x1 = df.loc[:, "X0":"ZSTANDDEV"]
        x2 = df.loc[:, 'RESULTANT']   
        cleaned_df = pd.concat([x1, x2], axis=1, join='inner')
        return cleaned_df 
    
    def scale_data(self, data, labels):
        """ Normalize the data using StandardScaler() function """
        scaler = StandardScaler().fit(data)
        data_normalized = scaler.transform(data)
        #convert labels into int
        activity_labels = []
        for l in labels: activity_labels.append(sorted(list(np.unique(labels))).index(l)) 
        return data_normalized, activity_labels
    
    def preprocess_data(self, dataframe):
        """ Preprocesse the data using balance(), clean(), and scale() functions """
        balanced_df = self.balance_data(dataframe)  
        activity_labels = balanced_df["ACTIVITY"]
        cleaned_df = self.clean_data(balanced_df)
        return self.scale_data(cleaned_df, activity_labels)

    def get_domains(self):
        indices = []
        domain_ids = []
        for idx in self.domains_idxs:
            indices_for_domain = np.where(np.array(self.patient_label_list) == idx)[0] 
            domain_ids.append(len(indices_for_domain) * [idx])
            indices.append(indices_for_domain) 
            
        domain_ids = np.concatenate(domain_ids)
        indices = np.concatenate(indices)
        return indices, domain_ids
    
    def __len__(self):
        """Return number of examples in the dataset"""
        return len(self.labels)   
    
    def __getitem__(self, index):
        domain_id = self.domain_ids[index]
        feature = self.features[index]
        label = self.labels[index]
        return feature, label, domain_id
        
        
# -------------------------------------------------------------------
'''
    Select domains for training or testing. 
    k = percentage of examples in the support set. q = percentage of examples in the test set.
'''
def domain_generator(dataset, domains_train_idxs, domains_test_idxs, k = 0.3, q = 0.7):
    domains = []
    domains_test = []

    # create each task = group = domain   
    for i in dataset.patients:
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
    
    support_size = min(len(Xs_sp), len(Xt_sp))
    query_size = min(len(Xs_qr), len(Xt_qr))
    
    #make the two domains with the same length
    Xs, ys, Xt, yt = process_domains(Xs, ys, Xt, yt)
            
    new_ys = torch.tensor([0] * len(ys))
    new_yt = torch.tensor([0] * len(yt))
    new_labels = random.sample(list(np.unique(ys)),len(np.unique(ys)))
    for i in range(len(np.unique(ys))):
        new_ys[np.where(ys == i)[0]] = new_labels[i]
        new_yt[np.where(yt == i)[0]] = new_labels[i]
        
    return (Xs[:support_size], new_ys[:support_size], Xs[support_size:], new_ys[support_size:], ds), (Xt[:support_size], new_yt[:support_size], Xt[support_size:], new_yt[support_size:], dt)

def process_domains(Xs, ys, Xt, yt):
    if len(Xs) < len(Xt):
        while len(Xs)!=len(Xt):
            idx = random.choice(range(len(Xt)))
            Xt = torch.cat([Xt[:idx], Xt[idx+1:]])
            yt = torch.cat([yt[:idx], yt[idx+1:]])
    elif len(Xs) > len(Xt):
        while len(Xs)!=len(Xt):
            idx = random.choice(range(len(Xs)))
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
