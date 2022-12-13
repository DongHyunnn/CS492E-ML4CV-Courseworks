  
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
from torch import optim

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Generator: 가짜 이미지를 생성합니다.
    # noise와 label을 결합하여 학습합니다..
class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.num_classes = params['num_classes'] # 클래스 수, 10
        self.nz = params['nz'] # 노이즈 수, 100
        self.input_size = params['input_size'] # (1,28,28)

        # noise와 label을 결합할 용도인 label embedding matrix를 생성합니다.
        self.label_emb = nn.Embedding(self.num_classes, self.num_classes)

        self.gen = nn.Sequential(
            nn.Linear(self.nz + self.num_classes, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,int(np.prod(self.input_size))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # noise와 label 결합
        gen_input = torch.cat((self.label_emb(labels),noise),-1)
        x = self.gen(gen_input)
        x = x.view(x.size(0), *self.input_size)
        return x

# Discriminator: 가짜 이미지와 진짜 이미지를 식별합니다.
class Discriminator(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.input_size = params['input_size']
        self.num_classes = params['num_classes']

        self.label_embedding = nn.Embedding(self.num_classes, self.num_classes)

        self.dis = nn.Sequential(
            nn.Linear(self.num_classes+int(np.prod(self.input_size)),512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Linear(512,512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Linear(512,1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # 이미지와 label 결합
        dis_input = torch.cat((img.view(img.size(0),-1),self.label_embedding(labels)),-1)

        x = self.dis(dis_input)
        return x
# 가중치 초기화
def initialize_weights(model):
    classname = model.__class__.__name__
    # fc layer
    if classname.find('Linear') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    # batchnorm
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

def sample_image(gen_model, n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = FloatTensor(np.random.normal(0, 1, (n_row ** 2, 100)))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = LongTensor(labels)
    gen_imgs = gen_model(z, labels)
    save_image(gen_imgs.data, "./GAN/images/%d.png" % batches_done, nrow=n_row, normalize=True)

def main():
    # Transformation 정의
    train_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5],[0.5]), #set color value to -1~1 scale
    ])
    
    # MNIST dataset 불러오기
    train_ds = datasets.MNIST('./data', train=True, transform=train_transform, download=True)


    # 데이터 로더 생성
    batch_size = 100
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # 파라미터 설정
    params = {'num_classes':10,
            'nz':100,
            'input_size':(1,28,28)}
    

    nz = params['nz']

    model_gen = Generator(params).to(device)
    model_dis = Discriminator(params).to(device)
    model_gen.load_state_dict(torch.load('/home/donghyeon/GAN/pretrained_weights/cgan_weights_gen.pt'))
    model_dis.load_state_dict(torch.load('/home/donghyeon/GAN/pretrained_weights/cgan_weights_dis.pt'))
    sample_image(gen_model = model_gen, n_row=10, batches_done=200)
    exit()

    model_gen.apply(initialize_weights)
    model_dis.apply(initialize_weights)


    # 손실 함수
    loss_func = nn.BCELoss() # E(logx * y) + (1-x)*log(1-y)


    lr = 2e-4
    beta1 = 0.5
    beta2 = 0.999

    # optimization
    opt_dis = optim.Adam(model_dis.parameters(), lr=lr, betas=(beta1,beta2))
    opt_gen = optim.Adam(model_gen.parameters(), lr=lr, betas=(beta1,beta2))


    num_epochs = 200

    loss_history={'gen':[],
                'dis':[]}

    # 학습
    batch_count = 0
    start_time = time.time()
    model_dis.train()
    model_gen.train()

    sample_image(gen_model = model_gen,n_row=10, batches_done=0)
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
            if epoch+1 % 10 == 0:
                sample_image(gen_model = model_gen, n_row=10, batches_done=epoch+1)
                
        
    # plot loss history
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(loss_history['gen'],label="G")
    plt.plot(loss_history['dis'],label="D")
    plt.xlabel('batch count')
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'Generator and Discriminator Loss During Training epoch{num_epochs} batch{batch_size}.png')
        
    # 가중치 저장
    path2models = './GAN/'
    os.makedirs(path2models, exist_ok=True)
    path2weights_gen = os.path.join(path2models, 'cgan_weights_gen.pt')
    path2weights_dis = os.path.join(path2models, 'cgan_weights_dis.pt')

    torch.save(model_gen.state_dict(), path2weights_gen)
    torch.save(model_dis.state_dict(), path2weights_dis)

    '''
        G_losses = []
        D_losses = []
                G_losses.append(g_loss.item())
                D_losses.append(d_loss.item())
    '''
if __name__ == "__main__":
    main()