  
import torchvision.datasets as datasets
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import time
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image
from torch.autograd import Variable
from cgan import Generator, initialize_weights
import os
from PIL import Image
from array import *
from random import shuffle

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')
  
# Transformation 정의
train_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.5],[0.5]),
])
  

def sample_image(n_img,train,labels):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    if train:   img_path = f'./GAN/synthesized_images/training-images'
    else:       img_path = f'./GAN/synthesized_images/test-images'
    # Sample noise
    z = FloatTensor(np.random.normal(0, 2, (n_img * 10, nz)))
    gen_imgs = model_gen(z, labels)
    data = list(zip(gen_imgs, labels.T))
    for name, (img,label) in enumerate(data):        
        save_image(img.data, f"{img_path}/{label.data}/img{name}.png" , normalize=True)
# 데이터 로더 생성
batch_size=64

# 파라미터 설정
params = {'num_classes':10,
          'nz':100,
          'input_size':(1,28,28)}

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

nz = params['nz']

model_gen = Generator(params).to(device)

model_gen.apply(initialize_weights)


model_gen.load_state_dict(torch.load('/home/donghyeon/GAN/pretrained_weights/cgan_weights_gen.pt'))
#sample_image(n_row=10, batches_done=200)

n_img = 6000   # number  of images per class
n_cls = 10  # number of classes ; default mnist = 10
# Get labels ranging from 0 to n_classes for n rows
labels = np.array([num for _ in range(n_img) for num in range(n_cls)])
labels = LongTensor(labels)
sample_image(n_img=6000,train=True,labels=labels)

labels = np.array([num for _ in range(1000) for num in range(n_cls)])
labels = LongTensor(labels)
sample_image(n_img=1000,train=False,labels=labels)


