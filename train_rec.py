from my_models import SimpleSeg, RecognizerOnPretrainedSegm
from dataset_classes import KaptchaDataset

import torch
import torch.nn.functional as f
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#inputs
print('Input dir name for recognition')
workdir = input()

print('Input csv name')
name_csv = input()

print('Input batch_size, num_epochs')
batch_size, num_epochs = map(int, input().split())

print('Input downsampling sizes (3), default = [24, 48, 96]')
downsampling_sizes = list(map(int, input().split()))

print('Input pred fc sizes (2), default = [96, 48]')
pred_fc_sizes = list(map(int, input().split()))

print('Input base model path')
base_path = input()

#modules
train_dataset = KaptchaDataset(workdir, 'train') 
test_dataset = KaptchaDataset(workdir, 'test')
validation_dataset = KaptchaDataset(workdir, 'validation')
base = torch.load(os.path.join('.', base_path), weights_only = False)
model = RecognizerOnPretrainedSegm(base, downsampling_sizes, pred_fc_sizes)

#data
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
validation_dataloader = DataLoader(validation_dataset, batch_size = batch_size, shuffle = True)

#metrics
def acc(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim = 1).squeeze()
    return (y_pred == y_true).sum() / (y_pred.shape[0] * y_pred.shape[1])

def abs_acc(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim = 1).squeeze()
    res = 0
    for i in range(y_pred.shape[0]):
        if (y_pred[i] == y_true[i]).sum() == 6:
            res += 1
    return res / 8   

#training loop
def another_train(model, criterion, train_dataloader, val_dataloader, num_epochs = 3, device='cpu'):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)
    #training puk
    for epoch in range(num_epochs):
        #training
        running_loss = 0
        running_acc = 0
        running_abs_acc = 0
        
        model.train()
        print('Training...')
        print(f'Epoch {epoch} / {num_epochs - 1}')

        for i, data in enumerate(tqdm(train_dataloader)):
            images, solutions = data
            
            images = images.to(device)
            solutions = solutions.to(device)
            
            preds = model(images)

            loss = criterion(preds, solutions)

            running_loss += loss.item()
            running_acc += acc(preds, solutions)
            running_abs_acc += abs_acc(preds, solutions)

            loss.backward()
            opt.step()
            opt.zero_grad()

        print(f'Training loss: {running_loss / len(train_dataloader)}')
        print(f'Training acc: {running_acc / len(train_dataloader)}')
        print(f'Training abs_acc: {running_abs_acc / len(train_dataloader)}')

        #validating
        running_loss = 0
        running_acc = 0
        running_abs_acc = 0
        
        model.eval()
        print('Validating...')
        for i, data in enumerate(tqdm(val_dataloader)):
            images, solutions = data
            
            images = images.to(device)
            solutions = solutions.to(device)

            with torch.no_grad():
                preds = model(images)
                
                loss = criterion(preds, solutions)
                
                running_loss += loss.item()
                running_acc += acc(preds, solutions)
                running_abs_acc += abs_acc(preds, solutions)

        print(f'Validating loss: {running_loss / len(val_dataloader)}')
        print(f'Valdiating acc: {running_acc / len(val_dataloader)}')
        print(f'Validating abs_acc: {running_abs_acc / len(val_dataloader)}')
    return

#testing loop
def test(model, test_dataloader, device = 'cpu'):
    running_acc = 0
    running_abs_acc = 0

    model.eval()

    print('Testing...')
    for i, data in enumerate(tqdm(test_dataloader)):
        images, solutions = data
        
        images = images.to(device)
        solutions = solutions.to(device)

        with torch.no_grad():
            preds = model(images)

            running_acc += acc(preds, solutions)
            running_abs_acc += abs_acc(preds, solutions)

    print(f'Testing acc: {running_acc / len(test_dataloader)}')
    print(f'Testing abs_acc: {running_abs_acc / len(test_dataloader)}')

#Train!
another_train(model, nn.CrossEntropyLoss(), train_dataloader, validation_dataloader, num_epochs=num_epochs)

#Test!
test(model, test_dataloader)

print('Done!')