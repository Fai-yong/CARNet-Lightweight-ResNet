import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import torch
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from torchinfo import summary

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x, return_feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        if return_feature:
            return out, y
        else:
            return y
    
    # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, out_list
    
    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        out = F.relu(self.bn1(self.conv1(x)))
        if layer_index == 1:
            out = self.layer1(out)
        elif layer_index == 2:
            out = self.layer1(out)
            out = self.layer2(out)
        elif layer_index == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        elif layer_index == 4:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)               
        return out

    # function to extact the penultimate features
    def penultimate_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        penultimate = self.layer4(out)
        out = F.avg_pool2d(penultimate, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, penultimate
    
def ResNet18(num_c):
    return ResNet(PreActBlock, [2,2,2,2], num_classes=num_c)

def ResNet34(num_c):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_c)

def ResNet50(num_c):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_c)

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


# Raw dataset
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_labels = {}

        # Create a mapping of class labels to integers
        self.class_labels = {}
        class_idx = 0

        # Iterate over sub-directories
        for class_dir in os.listdir(self.root_dir):
            class_dir_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_dir_path):
                self.class_labels[class_dir] = class_idx
                class_idx += 1

                # Iterate over images in the sub-directory
                for img_filename in os.listdir(class_dir_path):
                    if img_filename.endswith(".jpg"):
                        img_path = os.path.join(class_dir_path, img_filename)
                        self.images.append(img_path)
                        self.labels.append(self.class_labels[class_dir])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])        
        label = self.labels[idx]
        
        # 檢查有無灰階影像. 如有 -> 增加色彩通道
        if image.mode == "L":
            image = Image.merge("RGB", (image, image, image))
        # 檢查有無transform. 如有 -> transform (image)
        if self.transform:
            image = self.transform(image)
            
        return image, label
    

    # pytorch dataloader
def model_dataloder(weights, transform):
    """
    Will return 3 pytorch datalaoder
    """
    weights = weights
    
    data_folder = "E:/Kaggle"

    train_folder = data_folder + "/train"
    val_folder = data_folder + "/valid"
    test_folder = data_folder + "/test"
    
    # pytorch dataset
    train_dataset = ImageDataset(train_folder, transform = transform)
    val_dataset = ImageDataset(val_folder, transform = transform)
    test_dataset = ImageDataset(test_folder, transform = transform)
    
    # pytorch dataloader
 
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = 16, shuffle = True)
    val_dataloader = DataLoader(dataset = val_dataset, batch_size = 16, shuffle = False)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = 16, shuffle = False)
    
    return train_dataloader, val_dataloader, test_dataloader

# Train -> train_loss, train_acc
def train (model, dataloader, loss_fn, optimizer, device):
    train_loss, train_acc = 0, 0
    
    model.to(device)
    model.train()
    
    for batch, (x, y) in enumerate (dataloader):
        x, y = x.to(device), y.to(device)
        
        train_pred = model(x)
        
        loss = loss_fn(train_pred, y)
        train_loss = train_loss + loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_pred_label = torch.argmax(torch.softmax(train_pred, dim = 1), dim = 1)
        train_acc = train_acc + (train_pred_label == y).sum().item() / len(train_pred)
    
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    return train_loss, train_acc

# Val -> val_loss, val_acc
def val (model, dataloader, loss_fn, device):
    val_loss, val_acc = 0, 0
    
    model.to(device)
    model.eval()
    
    with torch.inference_mode():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            val_pred = model(x)
            
            loss = loss_fn(val_pred, y)
            val_loss = val_loss + loss.item()
            
            val_pred_label = torch.argmax(torch.softmax(val_pred, dim = 1), dim = 1)
            val_acc = val_acc + (val_pred_label == y).sum().item() / len(val_pred)
        
        val_loss = val_loss / len(dataloader)
        val_acc = val_acc / len(dataloader)
        
        return val_loss, val_acc
    
    # Training loop -> results dictionary
def training_loop(model, train_dataloader, val_dataloader, device, epochs, patience):
    # empty dict for restore results
    results = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    
    # hardcode loss_fn and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    
    # loop through epochs
    for epoch in range(epochs):
        train_loss, train_acc = train(model = model, 
                                      dataloader = train_dataloader,
                                      loss_fn = loss_fn,
                                      optimizer = optimizer,
                                      device = device)
        
        val_loss, val_acc = val(model = model,
                                dataloader = val_dataloader,
                                loss_fn = loss_fn,
                                device = device)
        
        # Append values to lists for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # print results for each epoch
        print(f"Epoch: {epoch+1}\n"
              f"Train loss: {train_loss:.4f} | Train accuracy: {(train_acc*100):.3f}%\n"
              f"Val loss: {val_loss:.4f} | Val accuracy: {(val_acc*100):.3f}%")
        
        # record results for each epoch
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        
        # calculate average "val_loss" for early_stopping
        mean_val_loss = np.mean(results["val_loss"])
        best_val_loss = float("inf")
        num_no_improvement = 0
        if np.mean(mean_val_loss > best_val_loss):
            best_val_loss = mean_val_loss
        
            model_state_dict = model.state_dict()
            best_model.load_state_dict(model_state_dict)
        else:
            num_no_improvement +=1
    
        if num_no_improvement == patience:
            break
    
    # Plotting
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.plot(train_accuracies, label="Train accuracy")
    plt.plot(val_accuracies, label="Val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return results

class ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(PreActBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(PreActBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(PreActBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(PreActBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y

# #下载预训练模型
# resnet_weight = torchvision.models.ResNet18_Weights.DEFAULT

resnet_model  = ResNet18(num_classes=100)



# for param in resnet_model.parameters():
#     param.requires_grad = False

# # Custom output layer
# # resnet_model.fc

# custom_fc = nn.Sequential(
#     nn.ReLU(),
#     nn.Dropout(p = 0.5),
#     nn.Linear(1000, 100))

# resnet_model.fc = nn.Sequential(
#     resnet_model.fc,
#     custom_fc
# )

# Check model info
summary(resnet_model, input_size = (1, 3, 244, 244), col_names = ["output_size", "num_params", "trainable"], col_width = 15)

Device
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cpu")
device = torch.device("cuda:0")

hardcode :
loss_fn -> CrossEntropyLoss
optimizer -> Adam(lr = 0.0005)

Data augmentation
resnet_weight.transforms()

resnet_transform = transforms.Compose([
    transforms.Resize(size = 232),
    transforms.ColorJitter(brightness = (0.8, 1.2)),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomRotation(degrees = 15),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

resnet_train_dataloader, resnet_val_dataloader, resnet_test_dataloader = model_dataloder(weights = None, 
                                                                                         transform = resnet_transform)
# Actual training ResNet model
resnet_results = training_loop(model = resnet_model,
                               train_dataloader = resnet_train_dataloader,
                               val_dataloader = resnet_val_dataloader,
                               device = device,
                               epochs = 30,
                               patience = 5
                              )

torch.save(resnet_model, 'resnet_model.pth')

# Plot training results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title("Loss")
plt.plot(resnet_results["train_loss"], label="Train loss")
plt.plot(resnet_results["val_loss"], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Accuracy")
plt.plot(resnet_results["train_acc"], label="Train accuracy")
plt.plot(resnet_results["val_acc"], label="Val accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()