import torch, random
import data, params
import math, numpy as np
from utils import accuracy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MLDG:
    def __init__(self, model, loss_fn, lr_inner, lr_adam=0.001, adapt_steps=1):
        self.model = model
        self.loss_fn = loss_fn
        self.lr_inner = lr_inner
        self.adapt_steps = adapt_steps

        self.theta = dict(self.model.named_parameters())
        meta_params = list(self.theta.values())
        self.optimizer = torch.optim.Adam(meta_params, lr=lr_adam)

    # -------------------------------------------------------------------
    def fit(self, domains, steps, beta=0.0001, print_output=True, multisource=False):
        for step in range(steps):
            inner_loss = 0.0
            if multisource:
                # Pick one random target domain for the evaluation
                t = random.choice(range(len(domains)))
                (Xt_sp, yt_sp, Xt_qr, yt_qr, dt) = domains[t]
                Xt, yt = torch.cat((Xt_sp, Xt_qr), dim=0), torch.cat((yt_sp, yt_qr), dim=0)

                for s in range(len(domains)):
                    if s != t:
                        (Xs_sp, ys_sp, Xs_qr, ys_qr, ds) = domains[s]
                        Xs, ys = torch.cat((Xs_sp, Xs_qr), dim=0), torch.cat((ys_sp, ys_qr), dim=0)

                        ## For FEMNIST
                        #Xs, ys, Xt, yt = data.process_domains(Xs, ys, Xt, yt)

                        Xs, ys, Xt, yt = Xs.to(DEVICE), ys.to(DEVICE), Xt.to(DEVICE), yt.to(DEVICE)

                        # Adapt with source domain
                        ys_pred = self.model(x=Xs)
                        inner_loss += self.loss_fn(ys_pred, ys)
            else:
                # Pick a random source domain and target domain from the meta-training set (domains)
                s, t = random.sample(range(len(domains)), 2)
                (Xs_sp, ys_sp, Xs_qr, ys_qr, ds), (Xt_sp, yt_sp, Xt_qr, yt_qr, dt) = domains[s], domains[t]
                Xs, ys = torch.cat((Xs_sp, Xs_qr), dim=0), torch.cat((ys_sp, ys_qr), dim=0)
                Xt, yt = torch.cat((Xt_sp, Xt_qr), dim=0), torch.cat((yt_sp, yt_qr), dim=0)

                Xs, ys, Xt, yt = Xs.to(DEVICE), ys.to(DEVICE), Xt.to(DEVICE), yt.to(DEVICE)

                ys_pred = self.model(x=Xs)
                inner_loss += self.loss_fn(ys_pred, ys)

            # Update with target domain and support set (labeled examples)
            yt_pred = self.model(x=Xt,
                                 meta_loss=inner_loss,
                                 meta_step_size=self.lr_inner,
                                 stop_gradient=False)
            outer_loss = self.loss_fn(yt_pred, yt)

            tot_loss = inner_loss + outer_loss * beta

            self.optimizer.zero_grad()
            tot_loss.backward()
            self.optimizer.step()

            if print_output:
                if (step + 1) % 50 == 0:
                    print(f"Step: {step + 1}, loss: {tot_loss.item():.5f}", end="\t\r")

        return self

# -------------------------------------------------------------------
# Evaluation
def evaluate_mldg(model, domains_test):
    test_accuracies = []
    for i in range(len(domains_test)):
        (Xt, yt, domain_t) = domains_test[i]
        Xt, yt = Xt.to(DEVICE), yt.to(DEVICE)

        yt_pred = model(Xt)

        ev = accuracy(yt_pred, yt)
        test_accuracies.append(ev)

    return test_accuracies
