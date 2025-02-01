from my_models import SimpleSeg
from dataset_classes import SegDataset

import torch
import torch.nn.functional as f
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
import os

#inputs
print('Input dir name for segmentation')
workdir = input()

print('Input csv name')
name_csv = input()

print('Input batch_size, val_size, num_epochs')
inpts = input().split()
batch_size, val_size, num_epochs = int(inpts[0]), float(inpts[1]), int(inpts[2])

print('Input smooth parameter (bool)')
smooth = input().lower()
smooth = True if smooth == 'true' else False

print('Input downsampling sizes (3), default = [16, 32, 64]')
downsampling_sizes = list(map(int, input().split()))

print('Input upsampling sizes (6), default = [64, 56, 48, 40, 32, 16]')
upsampling_sizes = list(map(int, input().split()))

#modules
dataset = SegDataset(workdir, name_csv)
model = SimpleSeg(smooth, downsampling_sizes, upsampling_sizes)

#data
data_len = dataset.__len__()
val_len = int(data_len * val_size)
train_dataset, val_dataset = random_split(dataset, [data_len - val_len, val_len])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

#metrics
def pixel_accuracy(preds, masks):
    masks = masks.squeeze()
    preds = torch.argmax(preds, dim = 1)
    correct_pixels = (preds == masks).sum()
    uncorrect_pixels = (preds != masks).sum()
    return correct_pixels / (correct_pixels + uncorrect_pixels)

def iou(preds, masks):
    preds = f.softmax(preds, dim=1)
    masks = masks.squeeze()

    mean_iou = 0
    for i in range(preds.shape[1]):
        cur_ch_preds = preds[:, i, :, :] > 0.5
        cur_ch_masks = masks == i
        
        ch_intersection = torch.logical_and(cur_ch_preds, cur_ch_masks).sum()
        ch_union = torch.logical_or(cur_ch_preds, cur_ch_masks).sum()
        
        mean_iou += ch_intersection / ch_union
    
    return mean_iou / preds.shape[1] 

#training loop
def train(model, criterion, train_dataloader, val_dataloader, num_epochs = 3, device='cpu'):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch} / {num_epochs - 1}')
        #training
        running_loss = 0
        running_acc = 0
        running_iou = 0

        model.train()
        print('Training...')
        for i, data in enumerate(tqdm(train_dataloader)):
            images, masks = data
            
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)

            loss = criterion(outputs, masks.squeeze())

            running_loss += loss.item()
            running_acc += pixel_accuracy(outputs, masks)
            running_iou += iou(outputs, masks)

            loss.backward()
            opt.step()
            opt.zero_grad()

        print(f'Training loss: {running_loss / len(train_dataloader)}')
        print(f'Training acc: {running_acc / len(train_dataloader)}')
        print(f'Training iou: {running_iou / len(train_dataloader)}')

        #validating
        running_loss = 0
        running_acc = 0
        running_iou = 0
        
        model.eval()
        print('Validating...')
        for i, data in enumerate(tqdm(val_dataloader)):
            images, masks = data
            
            images = images.to(device)
            masks = masks.to(device)

            with torch.no_grad():
                outputs = model(images)
                
                loss = criterion(outputs, masks.squeeze())
                
                running_loss += loss.item()
                running_acc += pixel_accuracy(outputs, masks)
                running_iou += iou(outputs, masks)

        print(f'Validating loss: {running_loss / len(val_dataloader)}')
        print(f'Valdiating acc: {running_acc / len(val_dataloader)}')
        print(f'Validating iou: {running_iou / len(val_dataloader)}')
    return

#Train!
train(model, nn.CrossEntropyLoss(), train_dataloader, val_dataloader, num_epochs=num_epochs)

#Save model weights
print('Saving model...')
torch.save(model.state_dict(), './semsegmodelweights.pt')
torch.save(model, './semsegmodel.pt')

print('Done!')