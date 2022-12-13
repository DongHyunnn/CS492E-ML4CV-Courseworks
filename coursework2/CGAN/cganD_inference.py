  
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from cgan import Discriminator, initialize_weights
import numpy as np 


cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')
  
# Transformation 정의
train_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.5],[0.5]),
])
  
# MNIST dataset 불러오기
test_ds = datasets.MNIST('./data', train=False, transform=train_transform, download=True)


# 데이터 로더 생성
batch_size=64
train_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

# 파라미터 설정
params = {'num_classes':10,
          'nz':100,
          'input_size':(1,28,28)}

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

model_dis = Discriminator(params).to(device)

model_dis.apply(initialize_weights)


model_dis.load_state_dict(torch.load('/home/donghyeon/GAN/cgan_weights_dis.pt'))


model_dis.eval()


with torch.no_grad():
    acc_list = []
    
    for (x, y) in train_dl:
        ba_si = x.shape[0]
        
        standard = torch.Tensor(ba_si,1).fill_(0.5).to(device) 
        yb_real = torch.Tensor(ba_si,1).fill_(1.0).to(device) 
        x = x.to(device)
        y = y.to(device)

        y_pred = model_dis(x, y) # 1 ==> real image
        succ = torch.count_nonzero(torch.greater_equal(y_pred,standard))
        acc = int(succ)/ba_si
        
        acc_list.append(acc)
        
print(f"Discriminator real accuracy : {np.mean(acc_list)*100} %")