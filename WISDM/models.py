import torch
from pytorch_revgrad.functional import revgrad
import math
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
from utils import LambdaLayer

# -------------------------------------------------------------------
''' DANN model ''' 
class DANNModel(torch.nn.Module):
    # -------------------------------------------------------------------
    def __init__(self, in_dim=43, alpha=1., n_classes=15):
        super().__init__()
        
        self.alpha = torch.tensor(alpha, requires_grad=False)
        
        self.features_extractor = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.BatchNorm1d(256, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.BatchNorm1d(256, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 32),
            torch.nn.ReLU(),
        )
        
        self.class_classifier = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.BatchNorm1d(64, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_classes)
        )

        self.domain_classifier = torch.nn.Sequential(
            torch.nn.Linear(32, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2),
        )
        
    # -------------------------------------------------------------------
    def forward(self, xa, xb, alpha):
        n = len(xa)
        x = torch.cat((xa, xb))
        x = self.features_extractor(x)
        out_cls = self.class_classifier(x)
        x_rev = revgrad(x, torch.tensor(alpha))
        out_dom = self.domain_classifier(x_rev)
        return out_cls[:n], out_cls[n:], out_dom[:n], out_dom[n:]


    
''' MMD model '''    
class MMDModel(torch.nn.Module):
    # -------------------------------------------------------------------
    def __init__(self, in_dim=43, n_classes=15):
        super().__init__()
        
        self.features_extractor = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.BatchNorm1d(256, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.BatchNorm1d(256, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 32),
            torch.nn.ReLU(),
        )
        
        self.class_classifier = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.BatchNorm1d(64, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_classes)
        )

    # -------------------------------------------------------------------
    def forward(self, xa, xb):
        n = len(xa)
        x = torch.cat((xa, xb))
        out_fe = self.features_extractor(x)
        out_cls = self.class_classifier(out_fe)
        return out_fe[:n], out_fe[n:], out_cls[:n], out_cls[n:]
    
    
# BASELINES -------------------------------------------------------------------
#-------------------------------------------------------------------
''' ARM-CML model '''
class ContextNet(torch.nn.Module):
    def __init__(self, in_dim=43, out_dim=1, hidden_dim=256):
        super(ContextNet, self).__init__()

        self.context_net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        out = self.context_net(x)
        return out
    
# -------------------------------------------------------------------
class ARMNet(torch.nn.Module):
    def __init__(self, in_dim=44, hidden_dim=64, n_classes=15):
        super(ARMNet, self).__init__()
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, n_classes)
                            )
        
    def forward(self, x):
        out = self.net(x)
        return out

#-------------------------------------------------------------------
''' 
    MLDG model 
    https://arxiv.org/abs/1710.03463
'''

def linear(inputs, weight, bias, meta_step_size=0.001, meta_loss=None, stop_gradient=False):
    if meta_loss is not None:

        if not stop_gradient:
            grad_weight = autograd.grad(meta_loss, weight, create_graph=True)[0]

            if bias is not None:
                grad_bias = autograd.grad(meta_loss, bias, create_graph=True)[0]
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        else:
            grad_weight = Variable(autograd.grad(meta_loss, weight, create_graph=True)[0].data, requires_grad=False)

            if bias is not None:
                grad_bias = Variable(autograd.grad(meta_loss, bias, create_graph=True)[0].data, requires_grad=False)
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        return F.linear(inputs,
                        weight - grad_weight * meta_step_size,
                        bias_adapt)
    else:
        return F.linear(inputs, weight, bias)


def conv2d(inputs, weight, bias, meta_step_size=0.001, stride=1, padding=0, dilation=1, groups=1, meta_loss=None,
           stop_gradient=False):
    if meta_loss is not None:

        if not stop_gradient:
            grad_weight = autograd.grad(meta_loss, weight, create_graph=True)[0]

            if bias is not None:
                grad_bias = autograd.grad(meta_loss, bias, create_graph=True)[0]
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        else:
            grad_weight = Variable(autograd.grad(meta_loss, weight, create_graph=True)[0].data,
                                   requires_grad=False)
            if bias is not None:
                grad_bias = Variable(autograd.grad(meta_loss, bias, create_graph=True)[0].data, requires_grad=False)
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        return F.conv2d(inputs,
                        weight - grad_weight * meta_step_size,
                        bias_adapt, stride,
                        padding,
                        dilation, groups)
    else:
        return F.conv2d(inputs, weight, bias, stride, padding, dilation, groups)


def relu(inputs):
    return F.threshold(inputs, 0, 0, inplace=True)


def maxpool(inputs, kernel_size, stride=None, padding=0):
    return F.max_pool2d(inputs, kernel_size, stride, padding=padding)

def batchnorm(inputs, running_mu, running_std):
    return F.batch_norm(inputs, running_mu, running_std, training=True, momentum=0.9)


class MLDGModel(torch.nn.Module):
    def __init__(self, in_dim=43, n_classes=15):
        super().__init__()

        self.fc1 = torch.nn.Linear(in_dim, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 32)
        self.fc4 = torch.nn.Linear(32, 64)
        self.fc5 = torch.nn.Linear(64, n_classes)

        # when you add the convolution and batch norm, below will be useful
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, meta_loss=None, meta_step_size=None, stop_gradient=False):
        x = linear(inputs=x,
                   weight=self.fc1.weight,
                   bias=self.fc1.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)
        x = F.relu(x, inplace=True)
        x = linear(inputs=x,
                   weight=self.fc2.weight,
                   bias=self.fc2.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)
        x = F.relu(x, inplace=True)
        x = linear(inputs=x,
                   weight=self.fc3.weight,
                   bias=self.fc3.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)
        x = F.relu(x, inplace=True)
        x = linear(inputs=x,
                   weight=self.fc4.weight,
                   bias=self.fc4.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)
        x = F.relu(x, inplace=True)
        x = linear(inputs=x,
                   weight=self.fc5.weight,
                   bias=self.fc5.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)
        return x

#-------------------------------------------------------------------
''' 
    DGER model 
    https://proceedings.neurips.cc/paper/2020/file/b98249b38337c5088bbc660d8f872d6a-Paper.pdf
'''

class MainModel(torch.nn.Module):
    def __init__(self, in_dim=43, n_classes=15):
        super().__init__()

        self.alpha = torch.tensor(1., requires_grad=False)

        self.features_extractor = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.BatchNorm1d(256, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.BatchNorm1d(256, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 32),
            torch.nn.ReLU(),
        )

        self.class_classifier = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.BatchNorm1d(64, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = self.features_extractor(x)
        out_class = self.class_classifier(x)
        return out_class, x

    def get_params(self, lr):
        return [{"params": self.parameters(), "lr": lr}]

class CC(torch.nn.Module):
    def __init__(self, n_domains=40, n_classes=15):
        super().__init__()
        self.classifier_list = torch.nn.ModuleList()
        for _ in range(n_domains):
            class_list = torch.nn.ModuleList()
            class_list.append(torch.nn.Linear(32, 64))
            class_list.append(torch.nn.BatchNorm1d(64, track_running_stats=False))
            class_list.append(torch.nn.ReLU())
            class_list.append(torch.nn.Linear(64, n_classes))
            self.classifier_list.append(torch.nn.Sequential(*class_list))

    def forward(self, x):
        output=[]
        for c, x_ in zip(self.classifier_list, x):
            output.append(c(x_))
        return output

    def get_params(self, lr):
        return [{"params": self.classifier_list.parameters(), "lr": lr}]

class CP(torch.nn.Module):
    def __init__(self, n_domains=40, n_classes=15):
        super().__init__()

        self.classifier_list = torch.nn.ModuleList()
        for _ in range(n_domains):
            class_list = torch.nn.ModuleList()
            class_list.append(torch.nn.Linear(32, 64))
            class_list.append(torch.nn.BatchNorm1d(64, track_running_stats=False))
            class_list.append(torch.nn.ReLU())
            class_list.append(torch.nn.Linear(64, n_classes))
            self.classifier_list.append(torch.nn.Sequential(*class_list))

    def forward(self, x):
        output = []
        for c, x_ in zip(self.classifier_list, x):
            x_ = revgrad(x_, torch.tensor(1.))
            output.append(c(x_))
        return output

    def get_params(self, lr):
        return [{"params": self.classifier_list.parameters(), "lr": lr}]

class DD(torch.nn.Module):
    def __init__(self, n_domains=40):
        super().__init__()

        self.domain_classifier = torch.nn.Sequential(
            torch.nn.Linear(32, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_domains),
        )

    def forward(self, x):
        x_rev = revgrad(x, torch.tensor(1.))
        x = self.domain_classifier(x_rev)
        return x

    def get_params(self, lr):
        return [{"params": self.domain_classifier.parameters(), "lr": lr}]


#-------------------------------------------------------------------
''' 
    MDAN model : Multi Source DANN
    https://www.cs.cmu.edu/~hzhao1/papers/NIPS2018/nips2018_main.pdf
'''
class MDANModel(torch.nn.Module):
    def __init__(self, in_dim=43, n_domains=40, n_classes=15):
        super().__init__()

        self.n_domains = n_domains

        self.features_extractor = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.BatchNorm1d(256, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.BatchNorm1d(256, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 32),
            torch.nn.ReLU(),
        )

        self.class_classifier = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.BatchNorm1d(64, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_classes)
        )

        self.domain_classifier = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(32, 256),
                                                                          torch.nn.ReLU(),
                                                                          torch.nn.Linear(256, 256),
                                                                          torch.nn.ReLU(),
                                                                          torch.nn.Linear(256, 256),
                                                                          torch.nn.ReLU(),
                                                                          torch.nn.Linear(256, 256),
                                                                          torch.nn.ReLU(),
                                                                          torch.nn.Linear(256, 2)) for _ in range(self.n_domains)])
        self.grls = [LambdaLayer(lambda x: revgrad(x, torch.tensor(1.))) for _ in range(self.n_domains)]

    def forward(self, sinputs, tinputs):
        """
                :param sinputs:     A list of k inputs from k source domains.
                :param tinputs:     Input from the target domain.
        """
        sx_list = [None]*len(sinputs)
        for i in range(len(sinputs)):
            sx_list[i] = self.features_extractor(sinputs[i])

        tx = self.features_extractor(tinputs)

        # Classification probabilities on k source domains.
        outcls = []
        for i in range(len(sinputs)):
            outcls.append(self.class_classifier(sx_list[i]))

        # Domain classification accuracies.
        outdom_s, outdom_t = [], []
        for i in range(len(sinputs)):
            outdom_s.append(self.domain_classifier[i](self.grls[i](sx_list[i])))
            outdom_t.append(self.domain_classifier[i](self.grls[i](tx)))

        return outcls, outdom_s, outdom_t

    def inference(self, inputs):
        x = self.features_extractor(inputs)
        out = self.class_classifier(x)
        return out