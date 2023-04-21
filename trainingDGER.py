import torch, random
import data, params, models
from utils import accuracy
from torch.nn import functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def all_models(in_dim, n_domains, n_classes):
    main_model = models.MainModel(in_dim, n_classes)
    dc = models.DD(n_domains)
    cc = models.CC(n_domains, n_classes)
    cp = models.CP(n_domains, n_classes)
    return main_model, dc, cc, cp


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_optim_and_scheduler(networks, lrs, lr_steps, gamma):
    if not isinstance(networks, list):
        networks = [networks]

    params = []
    for network, lr in zip(networks, lrs):
        if network is not None:
            params += network.get_params(lr)
    if not isinstance(lr_steps, list):
        lr_steps = [lr_steps, ]
    optimizer = torch.optim.SGD(params, weight_decay=.0005, momentum=.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=gamma)

    return optimizer, scheduler

class DGER:
    def __init__(self, n_domains, X_size):
        self.n_domains = n_domains
        self.X_size = X_size

        main_model, dis_model, c_model, cp_model = all_models(in_dim=params.input_dim, n_domains=n_domains, n_classes=params.n_classes)
        self.main_model = main_model.to(DEVICE)
        self.dis_model = dis_model.to(DEVICE)
        self.c_model = c_model.to(DEVICE)
        self.cp_model = cp_model.to(DEVICE)

        self.optimizer, self.scheduler = get_optim_and_scheduler([self.main_model, self.dis_model, self.c_model, self.cp_model],
                                                                 [0.001, params.lr_sgd_dd, params.lr_sgd_cc, params.lr_sgd_cc],
                                                                 lr_steps=60, gamma=0.1)

    def _compute_dis_loss(self, feature, domains):
        img_num_per_domain = [self.X_size]*self.n_domains
        if self.dis_model is not None:
            domain_logit = self.dis_model(feature)
            weight = [1.0 / img_num for img_num in img_num_per_domain]
            weight = torch.FloatTensor(weight).to(DEVICE)
            weight = weight / weight.sum() * self.n_domains
            domain_loss = F.cross_entropy(domain_logit, domains, weight=weight)
        else:
            domain_loss = torch.zeros(1, requires_grad=True).to(DEVICE)

        return domain_loss

    def _compute_cls_loss(self, model, feature, label, domain, mode="self"):
        if model is not None:
            feature_list = []
            label_list = []
            weight_list = []
            for i in range(self.n_domains):
                if mode == "self":
                    feature_list.append(feature[domain == i])
                    label_list.append(label[domain == i])
                else:
                    feature_list.append(feature[domain != i])
                    label_list.append(label[domain != i])
                weight = torch.zeros(params.n_classes).to(DEVICE)
                for j in range(params.n_classes):
                    weight[j] = 0 if (label_list[-1] == j).sum() == 0 else 1.0 / (label_list[-1] == j).sum().float()
                weight = weight / weight.sum()
                weight_list.append(weight)
            class_logit = model(feature_list)
            loss = 0
            for p, l, w in zip(class_logit, label_list, weight_list):
                if p is None:
                    continue
                loss += F.cross_entropy(p, l, weight=w) / self.n_domains
        else:
            loss = torch.zeros(1, requires_grad=True).to(DEVICE)

        return loss


    # -------------------------------------------------------------------
    def fit(self, domains_train, steps, print_output=True):
        for step in range(steps):
            if step < 10: #warmup_steps
                aux_weight = 0.01
                main_weight = 0.01
            else:
                aux_weight = 1
                main_weight = 1

            if self.n_domains < len(domains_train): #For CIFAR10C
                indexes = random.sample(range(len(domains_train)), self.n_domains)
            else:
                indexes = list(range(self.n_domains))

            data, labels, domains = [], [], []
            #for idx in range(self.n_domains):
            for i, idx in enumerate(indexes):
                (Xs_sp, ys_sp, Xs_qr, ys_qr, ds) = domains_train[idx]
                Xs, ys = torch.cat((Xs_sp, Xs_qr), dim=0), torch.cat((ys_sp, ys_qr), dim=0)
                data.append(Xs)
                labels.append(ys)
                domains.append(i*torch.ones(len(ys)).long())
            data = torch.cat(data, dim=0).to(DEVICE)
            labels = torch.cat(labels, dim=0).to(DEVICE)
            domains = torch.cat(domains, dim=0).to(DEVICE)

            set_requires_grad(self.main_model, False)
            set_requires_grad(self.c_model, True)
            _, feature = self.main_model(data)
            c_loss_self = self._compute_cls_loss(self.c_model, feature.detach(), labels, domains, mode="self")*aux_weight
            self.optimizer.zero_grad()
            c_loss_self.backward()
            self.optimizer.step()

            set_requires_grad([self.main_model, self.dis_model, self.c_model, self.cp_model], True)
            class_logit, feature = self.main_model(data)
            main_loss = F.cross_entropy(class_logit, labels) * main_weight

            dis_loss = self._compute_dis_loss(feature, domains) * aux_weight

            set_requires_grad(self.c_model, False)
            c_loss_others = self._compute_cls_loss(self.c_model, feature, labels, domains, mode="others") * aux_weight

            cp_loss = self._compute_cls_loss(self.cp_model, feature, labels, domains, mode="self") * aux_weight

            loss = dis_loss + c_loss_others + cp_loss + main_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss += c_loss_self

            if print_output:
                if (step + 1) % 50 == 0:
                    print(f"Step: {step + 1}, loss: {loss:.5f}", end="\t\r")

            del loss, main_loss, dis_loss, c_loss_self, c_loss_others, cp_loss

        return self

# -------------------------------------------------------------------
# Evaluation
def evaluate_dger(model, domains_test):
    test_accuracies = []
    for i in range(len(domains_test)):
        (Xt, yt, domain_t) = domains_test[i]
        Xt, yt = Xt.to(DEVICE), yt.to(DEVICE)

        yt_pred, _ = model(Xt)

        ev = accuracy(yt_pred, yt)
        test_accuracies.append(ev)

    return test_accuracies