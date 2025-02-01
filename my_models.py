import torch
from torch import nn
from collections import OrderedDict

class SimpleSeg(nn.Module):
    def __init__(self, smooth = True, downsampling_sizes = [16, 32, 64], upsampling_sizes = [64, 56, 48, 40, 32, 16]):
        super(SimpleSeg, self).__init__()
        self.smooth = smooth

        self.smoothing = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 3, kernel_size = (3,3), stride = (1,1), bias = False)),
            ('bn1', nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
            ('relu', nn.ReLU()),
            ('conv2', nn.Conv2d(3, 3, kernel_size = (5,5), stride = (1,1), bias = False)),
            ('bn2', nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
            ('relu', nn.ReLU())
        ])
        )

        self.downsampling = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, downsampling_sizes[0], kernel_size = (3,3), stride = (1,1), bias = False)),
            ('bn1', nn.BatchNorm2d(downsampling_sizes[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
            ('relu', nn.ReLU()),
            ('maxpool', nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))),
            ('conv2', nn.Conv2d(downsampling_sizes[0], downsampling_sizes[1], kernel_size = (3,3), stride = (1,1), bias = False)),
            ('bn2', nn.BatchNorm2d(downsampling_sizes[1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
            ('relu', nn.ReLU()),
            ('maxpool', nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding = (1,0))),
            ('conv3', nn.Conv2d(downsampling_sizes[1], downsampling_sizes[2], kernel_size = (3,3), stride = (1,1), bias = False)),
            ('bn3', nn.BatchNorm2d(downsampling_sizes[2], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
            ('relu', nn.ReLU()),
        ]))

        self.upsampling = nn.Sequential(OrderedDict([
            ('transconv1', nn.ConvTranspose2d(upsampling_sizes[0], upsampling_sizes[1], kernel_size = (3, 3), stride = (1,1), bias=False)),
            ('relu', nn.ReLU()),
            ('transconv2', nn.ConvTranspose2d(upsampling_sizes[1], upsampling_sizes[2], kernel_size = (2, 2), stride = (2,2), bias=False)),
            ('relu', nn.ReLU()),
            ('transconv3', nn.ConvTranspose2d(upsampling_sizes[2], upsampling_sizes[3], kernel_size = (3, 3), stride = (1,1), bias=False)),
            ('relu', nn.ReLU()),
            ('transconv4', nn.ConvTranspose2d(upsampling_sizes[3], upsampling_sizes[4], kernel_size = (3, 3), stride = (1,1), bias=False)),
            ('relu', nn.ReLU()),
            ('transconv5', nn.ConvTranspose2d(upsampling_sizes[4], upsampling_sizes[5], kernel_size = (5, 5), stride = (1,1), bias=False)),
            ('relu', nn.ReLU()),
            ('transconv6', nn.ConvTranspose2d(upsampling_sizes[5], 8, kernel_size = (5, 5), stride = (1,1), bias=False)),
            ('relu', nn.ReLU()),
            ('conv1', nn.Conv2d(8, 4, kernel_size = (3,3), stride = (1,1), padding=(0,1), bias=False)),
            ('bn1', nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
            ('relu', nn.ReLU()),
            ('conv2', nn.Conv2d(4, 2, kernel_size = (3,3), stride = (1,1), padding=(1,1), bias=False)),
        ]))

    def forward(self, image):
        output = image

        if self.smooth:
            output = self.smoothing(output)
        
        output = self.downsampling(output)
        output = self.upsampling(output)
        return output
    
class RecognizerOnPretrainedSegm(nn.Module):
    def __init__(self, base, downsampling_sizes = [24, 48, 96], pred_fc_sizes = [96, 48], fc = None):
        super(RecognizerOnPretrainedSegm, self).__init__()
        self.base = base
        for param in base.parameters():
            param.requires_grad = False

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.downsampling = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, downsampling_sizes[0], kernel_size = (3,3), stride = (1,1), bias = False, padding='same')),
            ('bn1', nn.BatchNorm2d(downsampling_sizes[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
            ('relu', nn.ReLU()),
            ('conv2', nn.Conv2d(downsampling_sizes[0], downsampling_sizes[1], kernel_size = (3,3), stride = (1,1), bias = False, groups=6)),
            ('bn2', nn.BatchNorm2d(downsampling_sizes[1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
            ('relu', nn.ReLU()),
            ('maxpool', nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))),
            ('conv3', nn.Conv2d(downsampling_sizes[1], downsampling_sizes[2], kernel_size = (3,3), stride = (1,1), padding = (1,0), bias = False, groups=6)),
            ('bn3', nn.BatchNorm2d(downsampling_sizes[2], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
            ('relu', nn.ReLU()),
            ('maxpool', nn.MaxPool2d(kernel_size=(1,2), stride=(1,4)))
        ]))
        #48x48
        self.pred_fc = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(pred_fc_sizes[0], pred_fc_sizes[1], kernel_size = (3,3), stride = (1,1), bias = False, groups=6)),
            ('bn1', nn.BatchNorm2d(pred_fc_sizes[1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
            ('relu', nn.ReLU()),
            ('maxpool', nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))),
            ('conv2', nn.Conv2d(pred_fc_sizes[1], 6, kernel_size = (3,3), stride = (1,1), bias = False, groups=6)),
            ('bn2', nn.BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
            ('relu', nn.ReLU()),
            ('maxpool', nn.MaxPool2d(kernel_size=(2,2), stride=(4,4)))
        ]))
        #8,6,10,10
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Sequential(OrderedDict([
                ('linear1', nn.Linear(600, 300)),
                ('relu', nn.ReLU()),
                ('linear2', nn.Linear(300, 60)),
            ]))

    def forward(self, images):
        output = self.softmax(self.base(images))[:,1,:,:].unsqueeze(1)
        output = self.downsampling(output)
        output = torch.flatten(self.pred_fc(output), start_dim = 1, end_dim = -1)
        output = self.fc(output).view(8, 10, 6)
        return output