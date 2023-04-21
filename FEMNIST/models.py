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
    def __init__(self, in_dim=1, alpha=1., n_classes=62):
        super().__init__()
        
        self.alpha = torch.tensor(alpha, requires_grad=False)
        
        self.features_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, 64, kernel_size=5, padding=2), 
            torch.nn.BatchNorm2d(64, track_running_stats=False), 
            torch.nn.ReLU(), 
            torch.nn.Conv2d(64, 64, kernel_size=5, padding=2), 
            torch.nn.BatchNorm2d(64, track_running_stats=False), 
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=5, padding=2), 
            torch.nn.BatchNorm2d(64, track_running_stats=False), 
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=5),
            torch.nn.BatchNorm2d(128, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 128, kernel_size=5),
            torch.nn.BatchNorm2d(128, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.AdaptiveAvgPool2d(1),  #output size = 1x1
        )        
        
        self.final = torch.nn.Sequential(
            torch.nn.Linear(128, 200),
            torch.nn.ReLU())
            
            
        self.class_classifier = torch.nn.Sequential(
            torch.nn.Linear(200, n_classes)
        )

        self.domain_classifier = torch.nn.Sequential(
            torch.nn.Linear(200, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2),
        )

    def forward(self, x, alpha):
        x = self.features_extractor(x)
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = self.final(x)
        out_cls = self.class_classifier(x)
        x_rev = revgrad(x, torch.tensor(alpha))
        out_dom = self.domain_classifier(x_rev)
        return [out_cls, out_dom]
    

# -------------------------------------------------------------------    
''' MMD model '''    
class MMDModel(torch.nn.Module):
    def __init__(self, in_dim=1, n_classes=62):
        super().__init__()
        
        self.features_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, 64, kernel_size=5, padding=2), 
            torch.nn.BatchNorm2d(64, track_running_stats=False), 
            torch.nn.ReLU(), 
            torch.nn.Conv2d(64, 64, kernel_size=5, padding=2), 
            torch.nn.BatchNorm2d(64, track_running_stats=False), 
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=5, padding=2), 
            torch.nn.BatchNorm2d(64, track_running_stats=False), 
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=5),
            torch.nn.BatchNorm2d(128, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 128, kernel_size=5),
            torch.nn.BatchNorm2d(128, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.AdaptiveAvgPool2d(1),  #output size = 1x1
        )
        self.final = torch.nn.Sequential(
            torch.nn.Linear(128, 200),
            torch.nn.ReLU())
        
        self.class_classifier = torch.nn.Sequential(
            torch.nn.Linear(200, n_classes)
        )
        
    def forward(self, x):
        x = self.features_extractor(x)
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        out_fe = self.final(x)
        out_cls = self.class_classifier(out_fe)
        return out_fe, out_cls
    
    
# BASELINES -------------------------------------------------------------------
# -------------------------------------------------------------------
''' ARM-CML model '''
class ContextNet(torch.nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, kernel_size=5):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (kernel_size - 1) // 2

        self.context_net = torch.nn.Sequential(
                                torch.nn.Conv2d(in_dim, hidden_dim, kernel_size, padding=padding),
                                torch.nn.BatchNorm2d(hidden_dim),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding),
                                torch.nn.BatchNorm2d(hidden_dim),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(hidden_dim, out_dim, kernel_size, padding=padding)
                            )


    def forward(self, x):
        out = self.context_net(x)
        return out
    
# -------------------------------------------------------------------
class ARMNet(torch.nn.Module):
    def __init__(self, in_dim=1, hidden_dim=128, n_classes=62):
        super(ARMNet, self).__init__()

        kernel_size = 5

        padding = (kernel_size - 1) // 2

        self.conv1 = torch.nn.Sequential(
                        torch.nn.Conv2d(in_dim, hidden_dim, kernel_size),
                        torch.nn.BatchNorm2d(hidden_dim),
                        torch.nn.ReLU(),
                        torch.nn.MaxPool2d(2)
                    )


        self.conv2 = torch.nn.Sequential(
                        torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size),
                        torch.nn.BatchNorm2d(hidden_dim),
                        torch.nn.ReLU(),
                        torch.nn.MaxPool2d(2)
                        )

        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.final = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, 200),
                    torch.nn.ReLU(),
                    torch.nn.Linear(200, n_classes)
                  )


    def forward(self, x):
        """Returns logit with shape (batch_size, num_classes)"""
        # x shape: batch_size, num_channels, w, h
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.adaptive_pool(out) # shape: batch_size, hidden_dim, 1, 1
        out = out.squeeze(dim=-1).squeeze(dim=-1) 
        out = self.final(out)

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

def adapt_avg_pool(inputs, output_size):
    return F.adaptive_max_pool2d(inputs, output_size)

def batchnorm(inputs, running_mu, running_std):
    return F.batch_norm(inputs, running_mu, running_std, training=True, momentum=0.9)

''' MLDG model '''
class MLDGModel(torch.nn.Module):
    def __init__(self, in_dim=1, n_classes=62):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_dim, 64, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(64, track_running_stats=False)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.bn2 = torch.nn.BatchNorm2d(64, track_running_stats=False)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.bn3 = torch.nn.BatchNorm2d(64, track_running_stats=False)
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=5)
        self.bn4 = torch.nn.BatchNorm2d(128, track_running_stats=False)
        self.conv5 = torch.nn.Conv2d(128, 128, kernel_size=5)
        self.bn5 = torch.nn.BatchNorm2d(128, track_running_stats=False)
        self.fc1 = torch.nn.Linear(128, 200)
        self.fc2 = torch.nn.Linear(200, n_classes)

        # when you add the convolution and batch norm, below will be useful
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, meta_loss=None, meta_step_size=None, stop_gradient=False):
        x = conv2d(inputs=x,
                   weight=self.conv1.weight,
                   bias=self.conv1.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)
        x = batchnorm(inputs=x,
                      running_mu=self.bn1.running_mean,
                      running_std=self.bn1.running_var)
        x = F.relu(x, inplace=True)
        x = conv2d(inputs=x,
                   weight=self.conv2.weight,
                   bias=self.conv2.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)
        x = batchnorm(inputs=x,
                      running_mu=self.bn2.running_mean,
                      running_std=self.bn2.running_var)
        x = F.relu(x, inplace=True)
        x = conv2d(inputs=x,
                   weight=self.conv3.weight,
                   bias=self.conv3.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)
        x = batchnorm(inputs=x,
                      running_mu=self.bn3.running_mean,
                      running_std=self.bn3.running_var)
        x = F.relu(x, inplace=True)
        x = conv2d(inputs=x,
                   weight=self.conv4.weight,
                   bias=self.conv4.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)
        x = batchnorm(inputs=x,
                      running_mu=self.bn4.running_mean,
                      running_std=self.bn4.running_var)
        x = F.relu(x, inplace=True)
        x = maxpool(inputs=x,
                   kernel_size=2,
                   stride=2,
                   padding=0)
        x = conv2d(inputs=x,
                   weight=self.conv5.weight,
                   bias=self.conv5.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)
        x = batchnorm(inputs=x,
                      running_mu=self.bn5.running_mean,
                      running_std=self.bn5.running_var)
        x = F.relu(x, inplace=True)
        x = maxpool(inputs=x,
                    kernel_size=2,
                    stride=2,
                    padding=0)
        x = adapt_avg_pool(inputs=x,
                           output_size=1)
        x = x.squeeze(dim=-1).squeeze(dim=-1)
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

        return x

#-------------------------------------------------------------------
''' 
    DGER model 
    https://proceedings.neurips.cc/paper/2020/file/b98249b38337c5088bbc660d8f872d6a-Paper.pdf
'''
class MainModel(torch.nn.Module):
    def __init__(self, in_dim=3, n_classes=10):
        super().__init__()

        self.alpha = torch.tensor(1., requires_grad=False)

        self.features_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, 64, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(64, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(64, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(64, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=5),
            torch.nn.BatchNorm2d(128, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 128, kernel_size=5),
            torch.nn.BatchNorm2d(128, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.AdaptiveAvgPool2d(1),  # output size = 1x1
        )
        self.final = torch.nn.Sequential(
            torch.nn.Linear(128, 200),
            torch.nn.ReLU())

        self.class_classifier = torch.nn.Sequential(
            torch.nn.Linear(200, n_classes)
        )

    def forward(self, x):
        x = self.features_extractor(x)
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = self.final(x)
        out_class = self.class_classifier(x)
        return out_class, x

    def get_params(self, lr):
        return [{"params": self.parameters(), "lr": lr}]

class CC(torch.nn.Module):
    def __init__(self, n_domains=297, n_classes=62):
        super().__init__()
        self.classifier_list = torch.nn.ModuleList()
        for _ in range(n_domains):
            self.classifier_list.append(torch.nn.Sequential(torch.nn.Linear(200, n_classes)))

    def forward(self, x):
        output=[]
        for c, x_ in zip(self.classifier_list, x):
            output.append(c(x_))
        return output

    def get_params(self, lr):
        return [{"params": self.classifier_list.parameters(), "lr": lr}]

class CP(torch.nn.Module):
    def __init__(self, n_domains=297, n_classes=62):
        super().__init__()

        self.classifier_list = torch.nn.ModuleList()
        for _ in range(n_domains):
            self.classifier_list.append(torch.nn.Sequential(torch.nn.Linear(200, n_classes)))

    def forward(self, x):
        output = []
        for c, x_ in zip(self.classifier_list, x):
            x_ = revgrad(x_, torch.tensor(1.))
            output.append(c(x_))
        return output

    def get_params(self, lr):
        return [{"params": self.classifier_list.parameters(), "lr": lr}]


class DD(torch.nn.Module):
    def __init__(self, n_domains):
        super().__init__()

        self.domain_classifier = torch.nn.Sequential(
            torch.nn.Linear(200, 256),
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
    def __init__(self, in_dim=3, n_domains=297, n_classes=62):
        super().__init__()

        self.n_domains = n_domains

        self.features_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, 64, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(64, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(64, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(64, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=5),
            torch.nn.BatchNorm2d(128, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 128, kernel_size=5),
            torch.nn.BatchNorm2d(128, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.AdaptiveAvgPool2d(1),  # output size = 1x1
        )

        self.final = torch.nn.Sequential(
            torch.nn.Linear(128, 200),
            torch.nn.ReLU())

        self.class_classifier = torch.nn.Sequential(
            torch.nn.Linear(200, n_classes)
        )

        self.domain_classifier = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(200, 256),
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
            sx = self.features_extractor(sinputs[i])
            sx = sx.squeeze(dim=-1).squeeze(dim=-1)
            sx_list[i] = self.final(sx)

        tx = self.features_extractor(tinputs)
        tx = tx.squeeze(dim=-1).squeeze(dim=-1)
        tx = self.final(tx)

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
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = self.final(x)
        out = self.class_classifier(x)
        return out