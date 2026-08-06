import torch.nn as nn
from .basic_layers.BatchNormalization import BatchNorm

# model from:
# S. Zhang, F. Ye, B. Wang and T. G. Habetler,
# "Few-Shot Bearing Fault Diagnosis Based on Model-Agnostic Meta-Learning,"
# in IEEE Transactions on Industry Applications, vol. 57, no. 5, pp. 4754-4764,
# Sept.-Oct. 2021, doi: 10.1109/TIA.2021.3091958.

class CNN2D4L_Backbone(nn.Module):
    def __init__(self, UseBatchNormalization: bool = True,
                 use_per_step_stats=False,
                 max_inner_steps=5, **kwargs):
        
        super(CNN2D4L_Backbone, self).__init__()
        
        self.use_bn = UseBatchNormalization

        # --- Block 1 ---
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = BatchNorm(self.conv1.out_channels, use_per_step_stats=use_per_step_stats,
                             max_inner_steps=max_inner_steps) if UseBatchNormalization else None
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # --- Block 2 ---
        self.conv2 = nn.Conv2d(self.conv1.out_channels, 64, kernel_size=3, padding=1)
        self.bn2 = BatchNorm(self.conv2.out_channels, use_per_step_stats=use_per_step_stats,
                             max_inner_steps=max_inner_steps) if UseBatchNormalization else None
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # --- Block 3 ---
        self.conv3 = nn.Conv2d(self.conv2.out_channels, 64, kernel_size=3, padding=1)
        self.bn3 = BatchNorm(self.conv3.out_channels, use_per_step_stats=use_per_step_stats,
                             max_inner_steps=max_inner_steps) if UseBatchNormalization else None
        self.relu3 = nn.ReLU(inplace=False)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # --- Block 4 ---
        self.conv4 = nn.Conv2d(self.conv3.out_channels, 64, kernel_size=3, padding=1)
        self.bn4 = BatchNorm(self.conv4.out_channels, use_per_step_stats=use_per_step_stats,
                             max_inner_steps=max_inner_steps) if UseBatchNormalization else None
        self.relu4 = nn.ReLU(inplace=False)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

    def forward(self, x, **kwargs):
        x = x.reshape(-1, 1, 64, 64)

        # Block 1
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x, kwargs)
        x = self.relu1(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x, kwargs)
        x = self.relu2(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        if self.bn3 is not None:
            x = self.bn3(x, kwargs)
        x = self.relu3(x)
        x = self.pool3(x)

        # Block 4
        x = self.conv4(x)
        if self.bn4 is not None:
            x = self.bn4(x, kwargs)
        x = self.relu4(x)
        x = self.pool4(x)

        return x
