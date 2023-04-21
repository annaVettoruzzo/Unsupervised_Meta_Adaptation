import torch

# UMA parameters
ADAPT_STEPS = 1 #adaptation steps at training time
training_steps = 10000
alpha_ = 1. #for GRL
sigma_ = 0.4 #for MMD
test_steps = 200 #adaptation steps at test time
loss = torch.nn.CrossEntropyLoss() #loss function
lr_sgd_cc = 0.1 #lr inner loop (label predictor)
lr_sgd_dd = 0.1 #lr inner loop (domain discriminator)
lr_adam = 0.001 #lr outer loop
batch_norm = True
num_comparison = None

# ARM-CML parameters
input_dim = 30
output_dim = 1
input_dim_arm = input_dim+1
hidden_dim_context = 256
hidden_dim_net = 64
support_size = 100
meta_batch_size = 2
learning_rate = 0.0001
weight_decay = 0
optimizer = "Adam"
momentum=0
epochsARM = 200
adapt_bn = 0
n_samples_test = 342

# Dataset parameters
k = 0.3
q = 0.7
n_classes = 19
nb_domains_tot = 8
nb_domains_train = 6
feature_reduction = True
extract_features = False #set to True the first time you run the code
img_dataset = False

seed = 123#0