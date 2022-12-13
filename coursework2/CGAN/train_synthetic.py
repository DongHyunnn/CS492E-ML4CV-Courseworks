  
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
from cgan import Generator, Discriminator,  initialize_weights
from torch import optim

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')
  
# Transformation 정의
train_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.5],[0.5]),
])
  


def sample_image(gen_model, n_row, batches_done, ratio):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = FloatTensor(np.random.normal(0, 2, (n_row ** 2, 100)))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = LongTensor(labels)
    gen_imgs = gen_model(z, labels)
    save_image(gen_imgs.data, f"./GAN/retrained/{ratio}/{batches_done}.png", nrow=n_row, normalize=True)


batch_size=64

# 파라미터 설정
params = {'num_classes':10,
          'nz':100,
          'input_size':(1,28,28)}

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

nz = params['nz']

# 손실 함수
loss_func = nn.BCELoss() # E(logx * y) + (1-x)*log(1-y)


lr = 2e-4
beta1 = 0.5
beta2 = 0.999
ratios = [10,20, 50, 100]

for ratio in ratios:
    print(f"Start training for ratio {ratio}")
    # Moderated MNIST dataset 불러오기
    train_ds = datasets.MNIST(f'./data/zipped_synthesized_images/{ratio}', train=True, transform=train_transform, download=False)

    # 데이터 로더 생성
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model_gen = Generator(params).to(device)
    model_dis = Discriminator(params).to(device)

    model_gen.apply(initialize_weights)
    model_dis.apply(initialize_weights)



    # optimization
    opt_dis = optim.Adam(model_dis.parameters(), lr=lr, betas=(beta1,beta2))
    opt_gen = optim.Adam(model_gen.parameters(), lr=lr, betas=(beta1,beta2))


    num_epochs = 200

    loss_history={'gen':[], 'dis':[]}

    # 학습
    batch_count = 0
    start_time = time.time()
    model_dis.train()
    model_gen.train()

    sample_image(gen_model=model_gen,n_row=10, ratio = ratio, batches_done=0)
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            ba_si = xb.shape[0]

            xb = xb.to(device)
            yb = yb.to(device)
            yb_real = torch.Tensor(ba_si,1).fill_(1.0).to(device) # real_label = 1
            yb_fake = torch.Tensor(ba_si,1).fill_(0.0).to(device) # fake_label = 0 

            # Genetator
            model_gen.zero_grad()
            noise = Variable(FloatTensor(np.random.normal(0, 1, (ba_si,nz)))).to(device)
            gen_label = Variable(LongTensor(np.random.randint(0, 10, ba_si))).to(device)

            # 가짜 이미지 생성
            out_gen = model_gen(noise, gen_label)

            # 가짜 이미지 판별
            out_dis = model_dis(out_gen, gen_label)

            loss_gen = loss_func(out_dis, yb_real)
            loss_gen.backward()
            opt_gen.step()

            # Discriminator
            model_dis.zero_grad()
            
            # 진짜 이미지 판별
            out_dis = model_dis(xb, yb)
            loss_real = loss_func(out_dis, yb_real)

            # 가짜 이미지 판별
            out_dis = model_dis(out_gen.detach(),gen_label)
            loss_fake = loss_func(out_dis,yb_fake)
            
            loss_dis = (loss_real + loss_fake) / 2
            loss_dis.backward()
            opt_dis.step()

            loss_history['gen'].append(loss_gen.item())
            loss_history['dis'].append(loss_dis.item())

            batch_count += 1
            if batch_count % 1000 == 0:
                print('Epoch: %.0f, G_Loss: %.6f, D_Loss: %.6f, time: %.2f min' %(epoch, loss_gen.item(), loss_dis.item(), (time.time()-start_time)/60))
            if (epoch+1) % 10 == 0:
                sample_image(gen_model=model_gen, n_row=10, ratio = ratio, batches_done=epoch+1)
                
        
    # plot loss history
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(loss_history['gen'],label="G")
    plt.plot(loss_history['dis'],label="D")
    plt.xlabel('batch count')
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'Trained w sr {ratio} data epoch{num_epochs} batch{batch_size}.png')
        
    # 가중치 저장
    path2models = './GAN/'
    os.makedirs(path2models, exist_ok=True)
    path2weights_gen = os.path.join(path2models, f'cgan_weights_gen_w_synr{ratio}.pt')
    path2weights_dis = os.path.join(path2models, f'cgan_weights_dis_w_synr{ratio}.pt')

    torch.save(model_gen.state_dict(), path2weights_gen)
    torch.save(model_dis.state_dict(), path2weights_dis)

'''
    G_losses = []
    D_losses = []
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
   '''
