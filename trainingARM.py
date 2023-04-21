import torch, random, copy, numpy as np, pickle
from collections import defaultdict
from utils import write_in_file

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ARM_CML(torch.nn.Module):

    def __init__(self, model, context_net, input_dim, loss_fn, learning_rate, weight_decay, optimizer, support_size, device, momentum=0, adapt_bn=0, img_dataset=1): 
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        
        self.context_net = context_net
        self.support_size = support_size
        self.input_dim = input_dim
        self.img_dataset = img_dataset
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.adapt_bn = adapt_bn

        params = list(self.model.parameters()) + list(self.context_net.parameters())
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif optimizer == "SGD":
            self.optimizer = torch.optim.SGD(params, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
    def predict(self, x):
        if self.img_dataset: batch_size, c, h, w = x.shape
        else: batch_size, w = x.shape

        if batch_size % self.support_size == 0:
            meta_batch_size, support_size = batch_size // self.support_size, self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size     

        if self.adapt_bn: #training batches are sampled from a single domain
            out = []
            for i in range(meta_batch_size):
                x_i = x[i*support_size:(i+1)*support_size]
                context_i = self.context_net(x_i)
                context_i = context_i.mean(dim=0).expand(support_size, -1, -1, -1)
                x_i = torch.cat([x_i, context_i], dim=1)
                out.append(self.model(x_i))
            return torch.cat(out)
        else:
            if self.img_dataset:
                context = self.context_net(x) # Shape: batch_size, channels, H, W
                context = context.reshape((meta_batch_size, support_size, self.input_dim, h, w))
                context = context.mean(dim=1) # Shape: meta_batch_size, self.input_dim
                context = torch.repeat_interleave(context, repeats=support_size, dim=0) # meta_batch_size*support_size, context_size
                x = torch.cat([x, context], dim=1)
            else:
                context = self.context_net(x.float())
                context = context.reshape((meta_batch_size, support_size))
                context = context.mean(dim=1)
                context = torch.repeat_interleave(context, repeats=support_size, dim=0)
                context = torch.unsqueeze(context, dim=-1)
                x = torch.cat([x, context], dim=1).float()        
            return self.model(x)
                                                     
    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def learn(self, x, labels, group_ids=None):

        self.train()
        # Forward
        logits = self.predict(x)
        loss = self.loss_fn(logits, labels)

        self.update(loss)

        stats = {'objective': loss.detach().item()}

        return logits, stats
    
    def get_acc(self, logits, labels):
        # Evaluate
        preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
        accuracy = np.mean(preds == labels.detach().cpu().numpy().reshape(-1))
        return accuracy
    
def run_epoch(algorithm, loader, train):

    epoch_labels = []
    epoch_logits = []
    epoch_group_ids = []

    for x, labels, group_ids in loader:

        # Put on GPU
        x = x.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward
        if train:
            logits, batch_stats = algorithm.learn(x, labels, group_ids)
            if logits is None: # DANN
                continue
        else:
            logits = algorithm.predict(x)

        epoch_labels.append(labels.to('cpu').clone().detach())
        epoch_logits.append(logits.to('cpu').clone().detach())
        epoch_group_ids.append(group_ids.to('cpu').clone().detach())

    return torch.cat(epoch_logits), torch.cat(epoch_labels), torch.cat(epoch_group_ids)

def train(algorithm, train_loader, test_loader, epochs = 200, print_output = False):
    
    history = defaultdict(list)

    # Train loop
    best_worst_case_acc = 0
    
    for epoch in range(epochs):
        #Train
        epoch_logits, epoch_labels, epoch_group_ids = run_epoch(algorithm, train_loader, train=True)
        
        #Evaluate
        worst_case_acc, average_case_acc, stats = eval_groupwise(algorithm, test_loader, epoch, split='test', n_samples_per_group=None)
        history["epoch"].append(epoch)
        history["worse_acc"].append(worst_case_acc)
        history["average_acc"].append(average_case_acc)
        
        if epoch % 10 == 0 and print_output:
            print(f"Epoch {epoch}")
            print(stats)
    return history

def get_group_iterator(loader, group, support_size, n_samples_per_group=None):
    example_ids = np.nonzero(loader.dataset.domain_ids == group)[0]
    example_ids = example_ids[np.random.permutation(len(example_ids))] # Shuffle example ids

    # Create batches
    batches = []
    X, Y, G = [], [], []
    counter = 0
    for i, idx in enumerate(example_ids):
        x, y, g = loader.dataset[idx]
        X.append(torch.tensor(x)); Y.append(y); G.append(g)
        if (i + 1) % support_size == 0:
            X, Y, G = torch.stack(X), torch.tensor(Y, dtype=torch.long), torch.tensor(G, dtype=torch.long)
            batches.append((X, Y, G))
            X, Y, G = [], [], []

        if n_samples_per_group is not None and i == (n_samples_per_group - 1):
            break
    if X:
        X, Y, G = torch.stack(X), torch.tensor(Y, dtype=torch.long), torch.tensor(G, dtype=torch.long)
        batches.append((X, Y, G))

    return batches

def eval_groupwise(algorithm, loader, epoch=None, split='test', n_samples_per_group=None):
    """ Test model on groups and log to wandb
        Separate script for femnist for speed."""

    groups = []
    accuracies = np.zeros(len(loader.dataset.domains_idxs))
    
    if algorithm.adapt_bn:
        algorithm.train()
    else:
        algorithm.eval()
    
    for i, group in enumerate(loader.dataset.domains_idxs):#tqdm(enumerate(loader.dataset.domains), desc='Evaluating', total=len(loader.dataset.domains)):
        counter = 0
        group_iterator = get_group_iterator(loader, group, algorithm.support_size, n_samples_per_group)

        logits, labels, group_ids = run_epoch(algorithm, group_iterator, train=False)
        preds = np.argmax(logits, axis=1)

        # Evaluate
        accuracy = np.mean((preds == labels).numpy())
        accuracies[i] = accuracy
        
    worst_case_acc = np.amin(accuracies)
    
    average_case_acc = np.mean(accuracies)

    stats = {
                f'worst_case_acc': worst_case_acc,
                f'average_acc': average_case_acc,
            }

    return worst_case_acc, average_case_acc, stats

def test(algorithm, loader, n_samples_per_group, save_output=""):
    worst_case_acc, average_case_acc, stats = eval_groupwise(algorithm, loader, epoch=None, split='test', n_samples_per_group=n_samples_per_group)
    if save_output: write_in_file(stats, save_output)
    return worst_case_acc, average_case_acc, stats