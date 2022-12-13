
from torchvision.models.inception import inception_v3
from torchvision.models import Inception_V3_Weights
import numpy as np
import torch
from torchvision.datasets import mnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import optim
from torch import nn
import time
from dataprocessing import preprocessing

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    epoch_acc = 0
    
    
    model.train()

    for (x, y) in iterator:

        model = model.to(device)
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):

    epoch_loss = 0
    epoch_acc = 0
    model = model.to(device)

    model.eval()
    
    with torch.no_grad():

        for (x, y) in iterator:
            
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item() 
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    ## set device to use
    device = torch.device('cuda' if cuda else 'cpu')
    if cuda : dtype = torch.cuda.FloatTensor
    else    : dtype = torch.FloatTensor

    # load inception_v3 model 
    pretrain=False
    if pretrain:    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    else :          model = inception_v3(init_weights=True)
    model.fc = nn.Linear(2048, 10)
    model.aux_logits = False
    model = model.to(device)
    
    #model.load_state_dict(torch.load('tut3-model.pt'))

    
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)   
    
    transform_train = transforms.Compose([ 
        transforms.Resize((299,299)),
        transforms.Grayscale(num_output_channels=3) ,
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
                            transforms.Resize((299,299)),
                            transforms.Grayscale(3),
                            transforms.ToTensor(),
    ])
    
    ratios = [0,10,20,50,100]
    for ratio in ratios:
        print(f"Current process: training inception v3 for {ratio}")
        
        # transfer learning to MNIST 
        
        ## set learning rate
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        ## criterion setting
        criterion = nn.CrossEntropyLoss()

        model = model.to(device)
        criterion = criterion.to(device)


        best_valid_loss = float('inf')

        ## preprocessing data
        train_data,valid_data= preprocessing(1234,ratio)

        ## creating iterator
        BATCH_SIZE = 200

        train_iterator = DataLoader(train_data,
                                    shuffle=True,
                                    batch_size=BATCH_SIZE)

        valid_iterator = DataLoader(valid_data,
                                    batch_size=BATCH_SIZE)

        ## train for few epochs
        EPOCHS = 3
        for epoch in range(EPOCHS):
            start_time = time.monotonic()

            train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
            valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f'Batch_{ratio}_mnist_inception_v3_weights.pt')

            end_time = time.monotonic()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        

        

        model.eval()
        
        print(f"Evaluating prediction accuracy of model trained with {ratio}_synthesized dataset")
        
        if ratio != 0:
            # load generated test dataset
            test_dataset = mnist.MNIST(root=f'data/zipped_synthesized_images/{ratio}/', train=False, transform=test_transform)
            
            batch_size = 100
            test_iterator = DataLoader(test_dataset, batch_size=batch_size)
            
            
            epoch_acc = 0
            for (x, y) in test_iterator:

                x = x.to(device)
                y = y.to(device)

                y_pred = model(x).detach()
                
                acc = calculate_accuracy(y_pred, y)
                epoch_acc += acc.item()

            print(f"\tPrediction Accuracy on Generated test set: {epoch_acc / len(test_iterator)*100:.2f}%")
                
                
        # load original MNIST test dataset
        test_dataset = mnist.MNIST(root='data/', train=False, transform=test_transform)
        
        batch_size = 100
        test_iterator = DataLoader(test_dataset)
        
        epoch_acc = 0
        for (x, y) in test_iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x).detach()
                     
            acc = calculate_accuracy(y_pred, y)
            epoch_acc += acc.item()

        print(f"\tPrediction Accuracy on MNIST test set: {epoch_acc / len(test_iterator)*100 :.2f}%")
            
            
        
            