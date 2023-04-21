from stateless import functional_call
import torch, random, copy, numpy as np, pandas as pd, pickle
from utils import zeros, ones, write_in_file
from collections import defaultdict
from mmd import rbf_mmd
import data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#sample a few data for testing
def sample_data(X, y, test_size, seed=None, y_ref=None):   
    if seed is not None: random.seed(seed)
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
class UMA():
    def __init__(self, model, loss_fn, lr_sgd_cc, lr_sgd_dd, lr_adam, adapt_steps=1, model_type = 'dann', adaptive_lr=False, sigma=None, batch_norm=False):
        self.model = model
        self.loss_fn = loss_fn
        self.lr_sgd_cc = torch.nn.Parameter(torch.tensor(lr_sgd_cc))
        self.lr_sgd_dd = torch.nn.Parameter(torch.tensor(lr_sgd_dd))
        self.adapt_steps = adapt_steps  
        self.model_type = model_type
        self.sigma = sigma #used only in mmd
        self.batch_norm = batch_norm
        
        self.theta = dict(self.model.named_parameters())
        meta_params = list(self.theta.values())
        if adaptive_lr and self.model_type =='dann': 
            meta_params.append(self.lr_sgd_cc)
            meta_params.append(self.lr_sgd_dd)
        if adaptive_lr and self.model_type == 'mmd':
            meta_params.append(self.lr_sgd_cc)

        self.optimizer = torch.optim.Adam(meta_params, lr = lr_adam) 

    # -------------------------------------------------------------------
    """
    Takes a list of parameters (params) and a sorce and target dataset (Xs, ys, Xt), and 
    returns the updated parameters after taking one gradient descent step
    """
    def adaptation_step(self, params, Xs_sp, Xs_qr, ys_sp, Xt_sp, Xt_qr, alpha):
        if self.model_type == 'dann':
            if self.batch_norm:
                out_cls_src, _, out_dom_src, _ = functional_call(self.model, params, (Xs_sp, Xs_qr, alpha))
                out_cls_tgt, _, out_dom_tgt, _ = functional_call(self.model, params, (Xt_sp, Xt_qr, alpha))
            else:    
                out_cls_src, out_dom_src = functional_call(self.model, params, (Xs_sp, alpha))
                out_cls_tgt, out_dom_tgt = functional_call(self.model, params, (Xt_sp, alpha))
            loss = self.loss_fn(out_cls_src, ys_sp) + self.loss_fn(out_dom_src, zeros(len(Xs_sp))) + self.loss_fn(out_dom_tgt, ones(len(Xt_sp)))
        elif self.model_type == 'mmd':
            if self.batch_norm:
                out_fe_src, _, out_cls_src, _ = functional_call(self.model, params, (Xs_sp, Xs_qr))
                out_fe_tgt, _, out_cls_tgt, _ = functional_call(self.model, params, (Xt_sp, Xt_qr))
            else:
                out_fe_src, out_cls_src = functional_call(self.model, params, Xs_sp)
                out_fe_tgt, out_cls_tgt = functional_call(self.model, params, Xt_sp)
            loss = self.loss_fn(out_cls_src, ys_sp) + rbf_mmd(out_fe_src, out_fe_tgt, self.sigma)
          
        grads = torch.autograd.grad(loss, params.values())
        dict_param = {}
        for (name, w), w_grad in zip(params.items(), grads):
            if "domain_classifier" in name:
                dict_param[name] = w - self.lr_sgd_dd * w_grad
            else:
                dict_param[name] = w - self.lr_sgd_cc * w_grad           
        return dict_param
    
    # -------------------------------------------------------------------
    """
    Returns the task specific parameters phi (after adapting theta with GD using one or multiple adaptation steps)
    """
    def get_adapted_parameters(self, Xs_sp, Xs_qr, ys_sp, Xt_sp, Xt_qr, alpha):
        phi = self.adaptation_step(self.theta, Xs_sp, Xs_qr, ys_sp, Xt_sp, Xt_qr, alpha)
        for _ in range(self.adapt_steps - 1):  #if we have more than one adaptation steps
            phi = self.adaptation_step(self.theta, Xs_sp, Xs_qr, ys_sp, Xt_sp, Xt_qr, alpha)
        return phi

    
    # -------------------------------------------------------------------
    """
    Takes a batch of data from the source task and the target task and trains the model parameters (theta) using MAML.
    """
    def fit_dann(self, domains, steps, alpha=None, clip_max_norm=None, print_output = False):
        for step in range(steps):
            p = step / steps
            alpha_ = 2. / (1. + np.exp(-10 * p)) - 1 if alpha is None else alpha
            
            #Pick a random source domain and target domain from the meta-training set (domains)
            s, t = random.sample(range(len(domains)), 2)
            (Xs_sp, ys_sp, Xs_qr, ys_qr, ds), (Xt_sp, yt_sp, Xt_qr, yt_qr, dt) = data.shuffle_labels(domains[s], domains[t])
            
            Xs_sp, ys_sp, Xs_qr, ys_qr= Xs_sp.to(DEVICE), ys_sp.to(DEVICE), Xs_qr.to(DEVICE), ys_qr.to(DEVICE)
            Xt_sp, yt_sp, Xt_qr, yt_qr= Xt_sp.to(DEVICE), yt_sp.to(DEVICE), Xt_qr.to(DEVICE), yt_qr.to(DEVICE)

            #Adaptation step (with support set) get the parameters adapted for this task
            phi = self.get_adapted_parameters(Xs_sp, Xs_qr, ys_sp, Xt_sp, Xt_qr, alpha_)
            
            #Loss outer loop (with query set)
            if self.batch_norm:
                _, out_cls_src, _, out_dom_src = functional_call(self.model, phi, (Xs_sp, Xs_qr, alpha_))
                _, out_cls_tgt, _, out_dom_tgt = functional_call(self.model, phi, (Xt_sp, Xt_qr, alpha_))
            else:
                out_cls_src, out_dom_src = functional_call(self.model, phi, (Xs_qr, alpha_))
                out_cls_tgt, out_dom_tgt = functional_call(self.model, phi, (Xt_qr, alpha_))
            
            lossout = self.loss_fn(out_cls_src, ys_qr) + self.loss_fn(out_cls_tgt, yt_qr) + self.loss_fn(out_dom_src, zeros(len(Xs_qr))) + self.loss_fn(out_dom_tgt, ones(len(Xt_qr)))

            #Optimize loss wrt theta
            self.optimizer.zero_grad()
            lossout.backward()
            if clip_max_norm is not None: # Gradients clipping to avoid the "exploiding gradients" problem
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_max_norm)
            self.optimizer.step()
                      
            if print_output:
                if (step+1)%50 == 0:
                    print(f"Step: {step+1}, alpha {alpha_:.5f}, lr_sgd_cc {self.lr_sgd_cc:.5f}, lr_sgd_dd {self.lr_sgd_dd:.5f}, loss: {lossout:.5f}", end="\t\r")
                    
        return self
    
    
    def fit_mmd(self, domains, steps, clip_max_norm=None, print_output = False):
        for step in range(steps):            
            alpha_ = None
            
            #Pick a random source domain and target domain from the meta-training set (domains)
            s, t = random.sample(range(len(domains)), 2)
            (Xs_sp, ys_sp, Xs_qr, ys_qr, ds), (Xt_sp, yt_sp, Xt_qr, yt_qr, dt) = data.shuffle_labels(domains[s], domains[t])
            
            Xs_sp, ys_sp, Xs_qr, ys_qr= Xs_sp.to(DEVICE), ys_sp.to(DEVICE), Xs_qr.to(DEVICE), ys_qr.to(DEVICE)
            Xt_sp, yt_sp, Xt_qr, yt_qr= Xt_sp.to(DEVICE), yt_sp.to(DEVICE), Xt_qr.to(DEVICE), yt_qr.to(DEVICE)

            #Adaptation step (with support set) get the parameters adapted for this task
            phi = self.get_adapted_parameters(Xs_sp, Xs_qr, ys_sp, Xt_sp, Xt_qr, alpha_)
            
            #Loss outer loop (with query set)
            if self.batch_norm:
                _, out_fe_src, _, out_cls_src = functional_call(self.model, phi, (Xs_sp, Xs_qr))
                _, out_fe_tgt, _, out_cls_tgt = functional_call(self.model, phi, (Xt_sp, Xt_qr))
            else:
                out_fe_src, out_cls_src = functional_call(self.model, phi, Xs_qr)
                out_fe_tgt, out_cls_tgt = functional_call(self.model, phi, Xt_qr)
            
            lossout = self.loss_fn(out_cls_src, ys_qr) + self.loss_fn(out_cls_tgt, yt_qr) + rbf_mmd(out_fe_src, out_fe_tgt, self.sigma)

            #Optimize loss wrt theta
            self.optimizer.zero_grad()
            lossout.backward()
            if clip_max_norm is not None: # Gradients clipping to avoid the "exploiding gradients" problem
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_max_norm)
            self.optimizer.step()
                      
            if print_output:
                if (step+1)%50 == 0:
                    print(f"Step: {step+1}, lr_sgd_cc {self.lr_sgd_cc:.5f}, sigma {self.sigma:.5f}, loss: {lossout:.5f}", end="\t\r")
                    
        return self
    
"""
Adapt parameters and test the model choosing one source domain at random.
"""
def adapt_evaluate_dann(model, loss_fn, lr_sgd_cc, lr_sgd_dd, Xs, ys, Xt, yt, steps, alpha = None, domain_eval_s = None, domain_eval_t = None, batch_norm=False, print_output=False):
    cmodel = copy.deepcopy(model) # To avoid modifying the original model
    optimizer = torch.optim.SGD([{'params': cmodel.features_extractor.parameters(), 'lr': lr_sgd_cc},
                                 {'params': cmodel.class_classifier.parameters(), 'lr': lr_sgd_cc},
                                 {'params': cmodel.domain_classifier.parameters(), 'lr': lr_sgd_dd}])                             
    history = defaultdict(list)
    
    totloss_classifier = []
    
    if len(Xs) != len(Xt): Xs, ys, Xt, yt = data.process_domains(Xs, ys, Xt, yt) 
    Xs, ys = Xs.to(DEVICE), ys.to(DEVICE)
    Xt, yt = Xt.to(DEVICE), yt.to(DEVICE)

    for step in range(steps):   
        p = step / steps
        alpha_ = 2. / (1. + np.exp(-10 * p)) - 1 if alpha is None else alpha
        
        # Adapt 
        if batch_norm:
            out_cls_src, _, out_dom_src, _ = cmodel(Xs, torch.zeros(0).to(DEVICE), alpha_)
            out_cls_tgt, _, out_dom_tgt, _ = cmodel(Xt, torch.zeros(0).to(DEVICE), alpha_) 
        else:
            out_cls_src, out_dom_src = cmodel(Xs, alpha_)
            out_cls_tgt, out_dom_tgt = cmodel(Xt, alpha_)

        loss_classifier = loss_fn(out_cls_src, ys)
        loss_domain = loss_fn(out_dom_src, zeros(len(Xs))) + loss_fn(out_dom_tgt, ones(len(Xt)))
                    
        loss = loss_classifier + loss_domain
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        totloss_classifier.append(loss_classifier.item())
        
        # Evaluate   
        if domain_eval_s is None: #evaluate on the same samples used for adaptation
            dys, dyt = zeros(len(Xs)), ones(len(Xt))
            acc_src = (out_cls_src.argmax(1) == ys).sum().item() / len(ys)
            acc_tgt = (out_cls_tgt.argmax(1) == yt).sum().item() / len(yt)
            acc_dom = ( (out_dom_src.argmax(1) == dys).sum().item() + (out_dom_tgt.argmax(1) == dyt).sum().item() ) / (len(dys) + len(dyt))
        else: #evaluate on all target domain
            (Xs_eval, ys_eval, _, _, _) = domain_eval_s
            (Xt_eval, yt_eval, _) = domain_eval_t
            if len(Xs_eval)!=len(Xt_eval): Xs_eval, ys_eval, Xt_eval, yt_eval = data.process_domains(Xs_eval, ys_eval, Xt_eval, yt_eval)
            with torch.no_grad():
                if batch_norm:
                    out_cls_src, _, out_dom_src, _ = cmodel(Xs_eval.to(DEVICE), torch.zeros(0).to(DEVICE), alpha_)
                    out_cls_tgt, _, out_dom_tgt, _ = cmodel(Xt_eval.to(DEVICE), torch.zeros(0).to(DEVICE), alpha_)
                else:
                    out_cls_src, out_dom_src = cmodel(Xs_eval.to(DEVICE), alpha_)
                    out_cls_tgt, out_dom_tgt = cmodel(Xt_eval.to(DEVICE), alpha_)                 
            dys, dyt = zeros(len(Xs_eval)), ones(len(Xt_eval))
            acc_src = (out_cls_src.argmax(1) == ys_eval.to(DEVICE)).sum().item() / len(ys_eval)
            acc_tgt = (out_cls_tgt.argmax(1) == yt_eval.to(DEVICE)).sum().item() / len(yt_eval)
            acc_dom = ( (out_dom_src.argmax(1) == dys).sum().item() + (out_dom_tgt.argmax(1) == dyt).sum().item() ) / (len(dys) + len(dyt))
        
        if (step+1)%10 == 0 and print_output:
            print(f"Step: {step+1}, alpha {alpha_:.5f}, acc_src {acc_src:.5f}, acc_tgt {acc_tgt:.5f}, acc_dom: {acc_dom:.5f}", end="\t\r")
 
        history["steps"].append(step)
        history["accuracy_src"].append(acc_src)
        history["accuracy_tgt"].append(acc_tgt)
        history["accuracy_dom"].append(acc_dom)
        history["prediction"].append(out_cls_tgt.detach().cpu().numpy())
         
    history["model"] = cmodel

    return history, totloss_classifier

"""
Adapt parameters and test the model choosing one source domain at random.
"""
def adapt_evaluate_mmd(model, loss_fn, lr_sgd_cc, Xs, ys, Xt, yt, steps, sigma=None, domain_eval_s = None, domain_eval_t = None, batch_norm=False, print_output=False):
    cmodel = copy.deepcopy(model) # To avoid modifying the original model
    optimizer = torch.optim.SGD(cmodel.parameters(), lr_sgd_cc)
        
    history = defaultdict(list)
    
    totloss_classifier = []

    if len(Xs)!=len(Xt): Xs, ys, Xt, yt = data.process_domains(Xs, ys, Xt, yt)
    Xs, ys = Xs.to(DEVICE), ys.to(DEVICE)
    Xt, yt = Xt.to(DEVICE), yt.to(DEVICE)

    for step in range(steps):   
        if batch_norm:
            out_fe_src,_,out_cls_src,_ = cmodel(Xs, torch.zeros(0).to(DEVICE))
            out_fe_tgt,_,out_cls_tgt,_ = cmodel(Xt, torch.zeros(0).to(DEVICE))
        else:
            out_fe_src, out_cls_src = cmodel(Xs)
            out_fe_tgt, out_cls_tgt = cmodel(Xt)

        loss_classifier = loss_fn(out_cls_src, ys)
        loss_domain = rbf_mmd(out_fe_src, out_fe_tgt, sigma)
                    
        loss = loss_classifier + loss_domain
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        totloss_classifier.append(loss_classifier.item())
        
        # Evaluate   
        if domain_eval_s is None: #evaluate on the same samples used for adaptation
            acc_src = (out_cls_src.argmax(1) == ys).sum().item() / len(ys)
            acc_tgt = (out_cls_tgt.argmax(1) == yt).sum().item() / len(yt)
        else: #evaluate on all target domain
            (Xs_eval, ys_eval, _, _, _) = domain_eval_s
            (Xt_eval, yt_eval, _) = domain_eval_t
            if len(Xs_eval)!=len(Xt_eval): Xs_eval, ys_eval, Xt_eval, yt_eval = data.process_domains(Xs_eval, ys_eval, Xt_eval, yt_eval)
            with torch.no_grad():
                if batch_norm:
                    out_fe_src,_,out_cls_src,_ = cmodel(Xs_eval.to(DEVICE),torch.zeros(0).to(DEVICE))
                    out_fe_tgt,_,out_cls_tgt,_ = cmodel(Xt_eval.to(DEVICE),torch.zeros(0).to(DEVICE)) 
                else:
                    out_fe_src, out_cls_src = cmodel(Xs_eval.to(DEVICE))
                    out_fe_tgt, out_cls_tgt = cmodel(Xt_eval.to(DEVICE))          
            dys, dyt = zeros(len(Xs_eval)), ones(len(Xt_eval))
            acc_src = (out_cls_src.argmax(1) == ys_eval.to(DEVICE)).sum().item() / len(ys_eval)
            acc_tgt = (out_cls_tgt.argmax(1) == yt_eval.to(DEVICE)).sum().item() / len(yt_eval)
        
        if (step+1)%10 == 0 and print_output:
            print(f"Step: {step+1}, acc_src {acc_src:.5f}, acc_tgt {acc_tgt:.5f}", end="\t\r")
 
        history["steps"].append(step)
        history["accuracy_src"].append(acc_src)
        history["accuracy_tgt"].append(acc_tgt)
        history["prediction"].append(out_cls_tgt.detach().cpu().numpy())
         
    history["model"] = cmodel

    return history, totloss_classifier

"""
Test the model on all test domains and compute the WORST and AVERAGE accuracy.
"""
def eval_domains(model, domains_train, domains_test, loss_fn, lr_sgd_cc, lr_sgd_dd, steps, alpha = None, sigma=None, model_type = 'dann', batch_norm=False, num_comparison=None, test_size = None, seed=None, save_output=""):  
    test_accuracies = [[]] * (len(domains_test))
    num_examples = np.zeros(len(domains_test))
    
    for i in range(len(domains_test)):
        (Xt, yt, domain_t) = domains_test[i]
        
        #sample a few data
        if test_size is not None: Xt, yt = sample_data(Xt, yt, test_size, seed)    
        
        num_examples[i] = len(yt)
        #select how many S domains to compare the target
        if num_comparison == None: sample_idx = range(len(domains_train))
        else: sample_idx = random.sample(range(len(domains_train)), num_comparison)    
        accuracies = [[]] * len(sample_idx)
        count = 0
        for j in sample_idx:
            (Xs_sp, ys_sp, _, _, domain_s) = domains_train[j]
            
            if test_size is not None: 
                Xs_sp, ys_sp = sample_data(Xs_sp, ys_sp, test_size, y_ref=yt)    
                if model_type == 'dann': history, _ = adapt_evaluate_dann(model, loss_fn, lr_sgd_cc, lr_sgd_dd, Xs_sp, ys_sp, Xt, yt, steps, alpha, domain_eval_s = domains_train[j], domain_eval_t = domains_test[i], batch_norm=batch_norm)
                elif model_type == 'mmd': history, _ = adapt_evaluate_mmd(model, loss_fn, lr_sgd_cc, Xs_sp, ys_sp, Xt, yt, steps, sigma, domain_eval_s = domains_train[j], domain_eval_t = domains_test[i], batch_norm=batch_norm)
            else:
                if model_type == 'dann': history, _ = adapt_evaluate_dann(model, loss_fn, lr_sgd_cc, lr_sgd_dd, Xs_sp, ys_sp, Xt, yt, steps, alpha, batch_norm=batch_norm)
                elif model_type == 'mmd': history, _ = adapt_evaluate_mmd(model, loss_fn, lr_sgd_cc, Xs_sp, ys_sp, Xt, yt, steps, sigma, batch_norm=batch_norm)
            
            accuracies[count] = history["accuracy_tgt"]
            count += 1
            
        #average accuracy for each test domain    
        test_accuracies[i] = np.mean(accuracies, axis=0)
    
    worst_case_acc = np.min(test_accuracies, axis=0)
    
    num_examples = np.array(num_examples)
    props = num_examples / num_examples.sum()

    average_case_acc = np.mean(test_accuracies, axis=0)
    
    stats = {
                f'average accuracy': average_case_acc,
                f'worst_case_accuracy': worst_case_acc,
                f'test_accuracies': test_accuracies,
            }       
    if save_output: write_in_file(stats, save_output)

    return worst_case_acc, average_case_acc, test_accuracies
