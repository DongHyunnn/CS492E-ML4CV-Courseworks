import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models

from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import time

def preprocessing(SEED,ratio):
    ## set ramdom seed to produce same result
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    transform_mnist2incepv3 = transforms.Compose([ 
        transforms.Resize((299,299)),
        transforms.Grayscale(num_output_channels=3) ,
        transforms.ToTensor(),
    ])
    path =f'./data/zipped_synthesized_images/{ratio}'
    if ratio == 0: path = './data'
    
    train_data = datasets.MNIST(path, train=True, transform=transform_mnist2incepv3, download=True)

    ## Create validation set                         
    VALID_RATIO = 0.8

    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = data.random_split(train_data,
                                            [n_train_examples, n_valid_examples])

    ## Ensure validation set uses the test transforms
    valid_data = copy.deepcopy(valid_data)
    valid_data.dataset.transform = transform_mnist2incepv3

    ## Check data sets has gone OK
    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')

    return train_data, valid_data

if __name__ == "__main__":
    preprocessing(1234,100)