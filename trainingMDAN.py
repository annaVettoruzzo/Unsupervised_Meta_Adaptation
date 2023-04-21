import torch, random, copy, numpy as np
from collections import defaultdict
from utils import accuracy, zeros, ones

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# sample a few data for testing
def sample_data(X, y, test_size, y_ref=None):
    if y_ref is not None:
        temp, idx = [], []
        for label in y_ref: temp.append(np.where(y == label)[0])
        idx.extend(random.choice(l) for l in temp)
    else:
        idx = random.sample(list(range(len(X))), test_size)
    X_new = [X[v] for v in idx]
    y_new = [y[v] for v in idx]

    return torch.stack(X_new), torch.tensor(y_new)

# -------------------------------------------------------------------
# Evaluation
def evaluate_mdan(model, domains_train, domains_test, n_src_domains, loss_fn, lr_adam, gamma=10.0, mu=1e-2, steps=1000, test_size=None, print_output=True):
    all_test_accuracies = [None] * len(domains_test)
    final_accuracy = []
    for t in range(len(domains_test)):
        (Xt, yt, domain_t) = domains_test[t]
        Xt, yt = Xt.to(DEVICE), yt.to(DEVICE)
        # sample a few data
        if test_size is not None: Xt, yt = sample_data(Xt, yt, test_size)

        # Use other domains as sources
        if n_src_domains < len(domains_train):  # For CIFAR10C
            indexes = random.sample(range(len(domains_train)), n_src_domains)
        else:
            indexes = list(range(n_src_domains))

        src_data, src_labels = [], []
        for i, idx in enumerate(indexes):
            (Xs_sp, ys_sp, Xs_qr, ys_qr, ds) = domains_train[i]
            Xs, ys = torch.cat((Xs_sp, Xs_qr), dim=0), torch.cat((ys_sp, ys_qr), dim=0)
            if test_size is not None: Xs, ys = sample_data(Xs, ys, test_size)
            src_data.append(Xs.to(DEVICE))
            src_labels.append(ys.to(DEVICE))

        cmodel = copy.deepcopy(model)  # To avoid modifying the original model
        optimizer = torch.optim.Adadelta(cmodel.parameters(), lr=lr_adam)

        test_accuracy = []
        for step in range(steps):

            outcls, outdom_s, outdom_t = cmodel(src_data, Xt)

            class_losses = torch.stack([loss_fn(outcls[j], src_labels[j]) for j in range(len(indexes))])
            dom_losses = torch.stack([loss_fn(outdom_s[j], zeros(len(outdom_s[j]))) + loss_fn(outdom_t[j], ones(len(outdom_t[j]))) for j in range(len(indexes))])

            loss = torch.log(torch.sum(torch.exp(gamma * (class_losses + mu * dom_losses)))) / gamma

            # Evaluation
            yt_pred = cmodel.inference(Xt)
            ev = accuracy(yt_pred, yt)
            test_accuracy.append(ev)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if print_output:
                if (step + 1) % 50 == 0:
                    print(f"Step: {step + 1}, loss: {loss:.5f}", end="\t\r")

        all_test_accuracies[t] = test_accuracy
        final_accuracy.append(test_accuracy[-1])
    return all_test_accuracies, final_accuracy



"""
Wrong implementation.

# -------------------------------------------------------------------
class MDAN():
    def __init__(self, model, loss_fn, lr_adam):
        self.model = model
        self.loss_fn = loss_fn

        self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=lr_adam)

    def fit(self, domains, n_domains, steps, gamma=10.0, mu=1e-2, print_output=True):
        for step in range(steps):
            tot_loss = 0

            #Sample target
            t = random.choice(range(len(domains)))
            (Xt_sp, yt_sp, Xt_qr, yt_qr, dt) = domains[t]
            Xt, yt = torch.cat((Xt_sp, Xt_qr), dim=0), torch.cat((yt_sp, yt_qr), dim=0)
            Xt, yt = Xt.to(DEVICE), yt.to(DEVICE)

            #Use other domains as sources
            if n_domains < len(domains): #For CIFAR10C
                indexes = random.sample(range(len(domains)), n_domains)
            else:
                indexes = list(range(n_domains))

            src_data, src_labels = [], []
            for i, idx in enumerate(indexes):
                if i != t:
                    (Xs_sp, ys_sp, Xs_qr, ys_qr, ds) = domains[i]
                    Xs, ys = torch.cat((Xs_sp, Xs_qr), dim=0), torch.cat((ys_sp, ys_qr), dim=0)
                    src_data.append(Xs.to(DEVICE))
                    src_labels.append(ys.to(DEVICE))

            outcls, outdom_s, outdom_t = self.model(src_data, Xt)

            class_losses = torch.stack([self.loss_fn(outcls[j], src_labels[j]) for j in range(len(indexes)-1)])
            dom_losses = torch.stack([self.loss_fn(outdom_s[j], zeros(len(outdom_s[j]))) + self.loss_fn(outdom_t[j], ones(len(outdom_t[j]))) for j in range(len(indexes)-1)])

            loss = torch.log(torch.sum(torch.exp(gamma * (class_losses + mu * dom_losses)))) / gamma

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if print_output:
                if (step + 1) % 50 == 0:
                    print(f"Step: {step + 1}, loss: {loss:.5f}", end="\t\r")

        return self
"""

