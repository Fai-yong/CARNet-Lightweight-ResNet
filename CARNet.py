import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch.nn.functional as F

from torch.autograd import Variable
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
import torchinfo
from torchinfo import summary

# 自定义注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes//ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(in_planes//ratio, in_planes, kernel_size=1)
    def forward(self, x):
        out = self.global_pool(x)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out * x

# 自定义块    
class GroupConvBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(GroupConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, groups=in_planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.attn = ChannelAttention(in_planes)
        self.conv2 = nn.Conv2d(in_planes, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.attn(out)
        out = self.conv2(out)
        return out

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

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)   

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
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

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        
        out = self.linear(out)
        return out
    
def ResNet18(num_classes=100):
    
    return ResNet(GroupConvBlock, [2,2,2,2], num_classes=num_classes)

# #下载预训练模型
# resnet_weight = torchvision.models.ResNet18_Weights.DEFAULT

resnet_model = ResNet18(num_classes=100)

# Check model info
summary(resnet_model, input_size = (1, 3, 244, 244), col_names = ["output_size", "num_params", "trainable"], col_width = 15)
# total_macs = summary.total_macs(resnet_model, input_size=(1, 3, 244, 244)) 
# total_macs = torchinfo.summary.total_macs(resnet_model, input_size=(1, 3, 244, 244))

# print(f"Total MACs: {total_macs/1e9:.2f} G")

device = "cuda" if torch.cuda.is_available() else "cpu"

start_time = time.time()

resnet_transform = transforms.Compose([
    transforms.Resize(size = 244),
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
                               epochs = 50,
                               patience = 5
                              )

end_time = time.time()
total_time = end_time - start_time

print(f"Total time: {total_time:.4f} seconds")

torch.save(resnet_model, 'resnet_model.pth')