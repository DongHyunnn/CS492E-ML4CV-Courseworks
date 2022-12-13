import os
import cv2
import numpy as np
import torchvision.datasets as datasets

path = './data'
dataset = datasets.MNIST(root=path, train=True)

for idx, (img, label) in enumerate(dataset):
    img.save(f'{path}/png_MNIST/training-images/{label}/{idx}.png')
    
dataset = datasets.MNIST(root=path, train=False)
for idx, (img, label) in enumerate(dataset):
    img.save(f'{path}/png_MNIST/test-images/{label}/{idx}.png')