
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

    # load inception_v3 model with pretrained weights
    model = inception_v3(init_weights=True)
    model.fc = nn.Linear(2048, 10)
    model.aux_logits = False
    model = model.to(device)
    
    model.load_state_dict(torch.load('tut3-model.pt'))

    
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

    # load original MNIST test dataset
    train_dataset = mnist.MNIST(root='data/', train=True, transform=transform_train)
    train_iterator = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

    testset = mnist.MNIST(root='data/', train=False, download=True, transform=test_transform)
    valid_iterator = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    
    # transfer learning to MNIST 
    
    ## set learning rate
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    ## set device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ## criterion setting
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)
    '''
    ## train for few epochs
    EPOCHS = 10

    best_valid_loss = float('inf')

    for epoch in range(EPOCHS):
        start_time = time.monotonic()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut3-model.pt')

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    '''
    print(f"Evaluation by Inception Score for synthesized images")
    model.eval()
    
    # load fully generated test dataset
    test_dataset = mnist.MNIST(root='data/64_zipped_synthesized_images/', train=True, transform=test_transform)
    
    batch_size = 100
    test_iterator = DataLoader(test_dataset, batch_size=batch_size)
    
    
    epoch_acc = 0
    IS = []
    for (x, y) in test_iterator:

        x = x.to(device)
        y = y.to(device)

        y_pred = model(x).detach()
        y_prob = softmax(y_pred.cpu().numpy()) #normalize to probability 0~1
        p_y = np.mean(y_prob,axis=0) #p(y)
        x = y_prob * np.log(y_prob/p_y) #p(y|x)log(p(y|x)/p(y))
        x = np.sum(x,axis=1) #KL
        x = np.mean(x,axis=0)
        IS.append(np.exp(x))
        
        acc = calculate_accuracy(y_pred, y)
        epoch_acc += acc.item()

    print(f"Prediction Accuracy: {epoch_acc / len(test_iterator)*100:.2f}%")
        
    print(f"Inception Score: {np.mean(IS):.5f}\n")
        
        
    # load original MNIST test dataset
    test_dataset = mnist.MNIST(root='data/', train=True, transform=ToTensor())
    
    batch_size = 100
    test_iterator = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"Evaluation by Inception Score for original MNIST images")
    model.eval()
    
    epoch_acc = 0
    IS = []
    for (x, y) in test_iterator:

        x = x.to(device)
        y = y.to(device)

        y_pred = model(x).detach()
        y_prob = softmax(y_pred.cpu().numpy()) #normalize to probability 0~1
        p_y = np.mean(y_prob,axis=0) #p(y)
        x = y_prob * np.log(y_prob/p_y) #p(y|x)log(p(y|x)/p(y))
        x = np.sum(x,axis=1) #KL
        x = np.mean(x,axis=0)
        IS.append(np.exp(x))
        
        acc = calculate_accuracy(y_pred, y)
        epoch_acc += acc.item()

    print(f"Prediction Accuracy: {epoch_acc / len(test_iterator)*100 :.2f}%")
        
    print(f"Inception Score: {np.mean(IS):.5f}\n")
        
    
        