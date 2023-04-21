import torch

# UMA parameters
ADAPT_STEPS = 1 #adaptation steps at training time
training_steps = 5600
alpha_ = 1. #for GRL
sigma_ = 0.4 #for MMD
test_steps = 200 #adaptation steps at test time
loss = torch.nn.CrossEntropyLoss() #loss function
lr_sgd_cc = 0.1 #lr inner loop (label predictor)
lr_sgd_dd = 0.1 #lr inner loop (domain discriminator)
lr_adam = 0.001 #lr outer loop
batch_norm = False
num_comparison = None

# ARM-CML parameters
input_dim = 1
output_dim = 1
input_dim_arm = input_dim+1
hidden_dim_context = 64
hidden_dim_net = 64
support_size = 50
meta_batch_size = 6
learning_rate = 0.0001
weight_decay = 0
optimizer = "Adam"
momentum=0
epochsARM = 200
adapt_bn = 0
n_samples_test = 252

# MatchDG params
img_c = 1
img_w = 28
img_h = 28
dataset_name="RotatedMNIST"

# Dataset parameters
k = 0.3
q = 0.7
n_classes = 10
nb_domains_tot = 14
nb_domains_train = 10
img_dataset = True

seed = 123