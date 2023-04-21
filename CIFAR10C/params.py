import torch

# UMA parameters
ADAPT_STEPS = 1 #adaptation steps at training time
training_steps = 37200
alpha_ = 1. #for GRL
sigma_ = 0.4 #for MMD
test_steps = 200 #adaptation steps at test time
loss = torch.nn.CrossEntropyLoss() #loss function
lr_sgd_cc = 0.1 #lr inner loop (label predictor)
lr_sgd_dd = 0.1 #lr inner loop (domain discriminator)
lr_adam = 0.001 #lr outer loop
batch_norm = False
num_comparison = 15

# ARM-CML parameters
input_dim = 3
output_dim = 3
input_dim_arm = input_dim+3
hidden_dim_context = 64
hidden_dim_net = 64
support_size = 100
meta_batch_size = 3
learning_rate = 1e-2
weight_decay = 1e-4
optimizer = "SGD"
momentum=0
epochsARM = 200
adapt_bn = 1
n_samples_test = 300

# Dataset parameters
k = 0.3
q = 0.7
n_classes = 10
nb_domains_tot = nb_domains_train = 56
img_dataset = True

seed = 0