import torch
import torch.nn as nn
import torch.nn.functional as F

class MediumCustomCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(MediumCustomCNN, self).__init__()

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # 1 -> 16 filters
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16 -> 32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32 -> 64
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 64 -> 128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 128 -> 256
        self.pool3 = nn.AdaptiveAvgPool2d((7, 7))  # Directly AdaptivePool after 3 blocks

        self.dropout = nn.Dropout(p=0.3)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Conv Block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        # Conv Block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        # Conv Block 3
        x = F.relu(self.conv5(x))
        x = self.pool3(x)

        x = self.dropout(x)

        # Flatten
        x = torch.flatten(x, 1)

        # FC Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        # Convolutional Block 1 (fewer filters)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Max Pooling
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Convolutional Block 2 (fewer filters)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Max Pooling
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


        self.dropout = nn.Dropout(p=0.4)  

        # Fully connected layers (reduce the size)
        self.fc1 = nn.Linear(256 * 7 * 7, 512)  # Flattened feature map size from Conv5
        self.fc2 = nn.Linear(512, 2)  # Output layer for binary classification (2 classes)

    def forward(self, x):
        # Apply convolutional layers with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = self.pool2(x)

        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = self.pool3(x)

        # Apply dropout for regularization
        x = self.dropout(x)

        # Flatten the feature maps for the fully connected layers
        x = x.view(-1, 256 * 7 * 7)  # Corrected the flattened size

        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        
        # Output layer (no activation function for raw logits, used for BCE loss)
        x = self.fc2(x)
        
        return x  # Output raw logits

# Instantiate the model
model = CustomCNN()

# Print the model architecture