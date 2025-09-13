import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x


class DifferentResNet186(nn.Module): #256
    def __init__(self, num_intervals, intermediate_features = 256, second_intermediate_features=256, dropout_prob = 0.5):
        super(DifferentResNet186, self).__init__()
        self.resnet18 = models.resnet18(pretrained = True)

        # for param in self.model.parameters():
        #     param.requires_grad = False

        self.resnet18.fc = Identity()
        
        self.fc1 = nn.Linear(512, intermediate_features)
        self.sigmoid1 = nn.Sigmoid() 
        self.batch_norm1 = nn.BatchNorm1d(intermediate_features)
        self.dropout1 = nn.Dropout(p=dropout_prob)

        self.fc2 = nn.Linear(intermediate_features, second_intermediate_features)
        self.sigmoid2 = nn.Sigmoid() 
        self.batch_norm2 = nn.BatchNorm1d(second_intermediate_features)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        
        self.survival = nn.Linear(second_intermediate_features, num_intervals)
        self.sigmoid3 = nn.Sigmoid() 

        
        
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        # Apply dropout before the FC layer
        
        x = self.fc1(x)
        x = self.sigmoid1(x)
        x = self.batch_norm1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.sigmoid2(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)

        x = self.survival(x)
        x = self.sigmoid3(x)

        return x
    
class Resnet502(nn.Module):
    def __init__(self, num_intervals, intermediate_features = 2048, second_intermediate_features=512, dropout_prob = 0.5):
        super(Resnet502, self).__init__()
        self.resnet50 = models.resnet50(pretrained = True)

        # for param in self.model.parameters():
        #     param.requires_grad = False

        self.resnet50.fc = Identity()
        
        self.fc = nn.Linear(2048, intermediate_features)
        self.sigmoid1 = nn.Sigmoid() 
        self.batch_norm = nn.BatchNorm1d(intermediate_features)
        self.dropout = nn.Dropout(p=dropout_prob)

        self.fc2 = nn.Linear(intermediate_features, second_intermediate_features)
        self.sigmoid2 = nn.Sigmoid() 
        self.batch_norm2 = nn.BatchNorm1d(second_intermediate_features)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        
        self.survival = nn.Linear(second_intermediate_features, num_intervals)
        self.sigmoid3 = nn.Sigmoid() 

        
        
    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)
        # Apply dropout before the FC layer
        
        x = self.fc(x)
        x = self.sigmoid1(x)
        x = self.batch_norm(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.sigmoid2(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)

        x = self.survival(x)
        x = self.sigmoid3(x)

        return x

class DifferentResNet187(nn.Module):
    def __init__(self, num_intervals = 0, intermediate_features = 512, dropout_prob = 0.7):
        super(DifferentResNet187, self).__init__()
        self.resnet18 = models.resnet18(pretrained = True)

        # for param in self.model.parameters():
        #     param.requires_grad = False

        self.resnet18.fc = Identity()
    

        
        
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        # Apply dropout before the FC layer
        
        x = self.fc(x)
        return x

class GoodUmapandGraph(nn.Module):
    def __init__(self, num_intervals, intermediate_features = 512, dropout_prob = 0.5):
        super(GoodUmapandGraph, self).__init__()
        self.resnet18 = models.resnet18(pretrained = True)

        self.resnet18.fc = Identity()
        
        self.fc = nn.Linear(512, intermediate_features)
        self.sigmoid1 = nn.Sigmoid() 
        self.batch_norm = nn.BatchNorm1d(intermediate_features)
        self.dropout = nn.Dropout(p=dropout_prob)
        

        self.survival = nn.Linear(intermediate_features, num_intervals)
        self.sigmoid2 = nn.Sigmoid() 

        
        
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        # Apply dropout before the FC layer
        
        x = self.fc(x)
        x = self.sigmoid1(x)
        x = self.batch_norm(x)
        x = self.dropout(x)

        x = self.survival(x)
        x = self.sigmoid2(x)

        return x
    


#512 -not oversample- 1 conv wroekd
class DifferentResNet18(nn.Module):
    def __init__(self, num_intervals, intermediate_features = 1024, second_intermediate_features = 512, dropout_prob=0.5):
        super(DifferentResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained = True)
        """
        for param in self.resnet18.parameters():
            param.requires_grad = False
        
        for param in self.resnet18.fc.parameters():
            param.requires_grad = True
        """
        conv1 = nn.Conv2d(in_channels=256,  # Input channels from Layer 3
                  out_channels=256,  # Number of filters
                  kernel_size=3,     # 3x3 kernel
                  stride=2,          # Stride of 2 (downsampling)
                  padding=1,         # Padding of 1 to maintain spatial dimensions
                  bias=False)        # Typically bias=False when followed by BatchNorm
        bn1 = nn.BatchNorm2d(conv1.out_channels)
        relu = nn.ReLU(inplace=True)

        conv2 = nn.Conv2d(in_channels=512,  # Input channels from Layer 3
                  out_channels=512,  # Number of filters
                  kernel_size=3,     # 3x3 kernel
                  stride=1,          # Stride of 2 (downsampling)
                  padding=1,         # Padding of 1 to maintain spatial dimensions
                  bias=False)        # Typically bias=False when followed by BatchNorm
        bn2 = nn.BatchNorm2d(512)

        layer4_block1 = nn.Sequential(conv1, bn1, relu)

        self.resnet18.layer4 = layer4_block1
        
        self.fc = nn.Linear(conv1.out_channels, intermediate_features)
        #nn.init.constant_(self.fc.weight, 0.5)
        self.batch_norm = nn.BatchNorm1d(intermediate_features)
        self.dropout = nn.Dropout(p=dropout_prob)

        self.survival = nn.Linear(intermediate_features, num_intervals)
        self.sigmoid = nn.Sigmoid() 

        self.resnet18.fc = Identity()
        
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply dropout before the FC layer
        
        x = self.fc(x)
        x = self.batch_norm(x)
        x = self.dropout(x)

        x = self.survival(x)
        x = self.sigmoid(x)

        return x

#512 -not oversample- 1 conv wroekd
class DifferentResNet182(nn.Module):
    def __init__(self, num_intervals, intermediate_features = 128, second_intermediate_features = 512, dropout_prob=0.5):
        super(DifferentResNet182, self).__init__()
        self.resnet18 = models.resnet18(pretrained = True)
        """
        for param in self.resnet18.parameters():
            param.requires_grad = False

        for param in self.resnet18.fc.parameters():
            param.requires_grad = True
        """
        conv1 = nn.Conv2d(in_channels=256,  # Input channels from Layer 3
                  out_channels=512,  # Number of filters
                  kernel_size=3,     # 3x3 kernel
                  stride=2,          # Stride of 2 (downsampling)
                  padding=1,         # Padding of 1 to maintain spatial dimensions
                  bias=False)        # Typically bias=False when followed by BatchNorm
        bn1 = nn.BatchNorm2d(conv1.out_channels)
        relu = nn.ReLU(inplace=True)

        conv2 = nn.Conv2d(in_channels=512,  # Input channels from Layer 3
                  out_channels=512,  # Number of filters
                  kernel_size=3,     # 3x3 kernel
                  stride=1,          # Stride of 2 (downsampling)
                  padding=1,         # Padding of 1 to maintain spatial dimensions
                  bias=False)        # Typically bias=False when followed by BatchNorm
        bn2 = nn.BatchNorm2d(512)

        layer4_block1 = nn.Sequential(conv1, bn1, relu)

        self.resnet18.layer4 = layer4_block1
        
        self.fc = nn.Linear(512, intermediate_features)
        #nn.init.constant_(self.fc.weight, 0.5)
        self.batch_norm = nn.BatchNorm1d(intermediate_features)
        self.dropout = nn.Dropout(p=dropout_prob)

        self.survival = nn.Linear(intermediate_features, num_intervals)
        self.sigmoid = nn.Sigmoid() 

        self.resnet18.fc = Identity()
        
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply dropout before the FC layer
        
        x = self.fc(x)
        x = self.batch_norm(x)
        x = self.dropout(x)

        x = self.survival(x)
        x = self.sigmoid(x)

        return x

#512 -not oversample- 1 conv wroekd
class DifferentResNet183(nn.Module):
    def __init__(self, num_intervals, intermediate_features = 128, dropout_prob=0.5):
        super(DifferentResNet183, self).__init__()
        self.resnet18 = models.resnet18(pretrained = True)
        conv1 = nn.Conv2d(in_channels=256,  # Input channels from Layer 3
                  out_channels=512,  # Number of filters
                  kernel_size=3,     # 3x3 kernel
                  stride=2,          # Stride of 2 (downsampling)
                  padding=1,         # Padding of 1 to maintain spatial dimensions
                  bias=False)        # Typically bias=False when followed by BatchNorm
        bn1 = nn.BatchNorm2d(conv1.out_channels)
        sigmoid = nn.Sigmoid()

        layer4_block1 = nn.Sequential(conv1, bn1, sigmoid)

        self.resnet18.layer4 = layer4_block1
        
        self.fc = nn.Linear(conv1.out_channels, intermediate_features)
        self.sigmoid1 = nn.Sigmoid() 
        self.batch_norm = nn.BatchNorm1d(intermediate_features)
        self.dropout = nn.Dropout(p=dropout_prob)
        

        self.survival = nn.Linear(intermediate_features, num_intervals)
        self.sigmoid2 = nn.Sigmoid() 

        self.resnet18.fc = Identity()
        
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply dropout before the FC layer
        
        x = self.fc(x)
        x = self.sigmoid1(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        
        x = self.survival(x)
        x = self.sigmoid2(x)

        return x

#64 features before
class DifferentResNet184(nn.Module):
    def __init__(self, num_intervals, intermediate_features = 1024, dropout_prob=0.5):
        super(DifferentResNet184, self).__init__()
        self.resnet18 = models.resnet18(pretrained = True)

        #self.resnet18.layer4 = Identity()
        self.resnet18.fc = Identity()
        
        self.fc = nn.Linear(512, intermediate_features)
        self.sigmoid1 = nn.Sigmoid() 
        self.batch_norm = nn.BatchNorm1d(intermediate_features)
        self.dropout = nn.Dropout(p=dropout_prob)
        

        self.survival = nn.Linear(intermediate_features, num_intervals)
        self.sigmoid2 = nn.Sigmoid() 

        
        
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply dropout before the FC layer
        
        x = self.fc(x)
        x = self.sigmoid1(x)
        x = self.batch_norm(x)
        x = self.dropout(x)

        x = self.survival(x)
        x = self.sigmoid2(x)

        return x


class DifferentResNet185(nn.Module):
    def __init__(self, num_intervals, intermediate_features = 400, dropout_prob=0.5):
        super(DifferentResNet185, self).__init__()
        self.resnet18 = models.resnet18(pretrained = True)

        self.resnet18.layer4 = Identity()
        self.resnet18.fc = Identity()
        
        self.fc = nn.Linear(256, intermediate_features)
        self.sigmoid1 = nn.Sigmoid() 
        self.batch_norm = nn.BatchNorm1d(intermediate_features)
        self.dropout = nn.Dropout(p=dropout_prob)
        

        self.survival = nn.Linear(intermediate_features, num_intervals)
        self.sigmoid2 = nn.Sigmoid() 

        
        
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)

        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply dropout before the FC layer
        
        x = self.fc(x)
        x = self.sigmoid1(x)
        x = self.batch_norm(x)
        x = self.dropout(x)

        x = self.survival(x)
        x = self.sigmoid2(x)

        return x