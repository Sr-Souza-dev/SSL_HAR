import torch.nn as nn
import torch.nn.functional as F

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=12, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=12, out_channels=24, kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=24, out_channels=48, kernel_size=2)
    def forward(self, x):
        x = F.leaky_relu(input=self.conv1(x), negative_slope=0.01)
        x = self.pool(x)
        x = F.leaky_relu(input=self.conv2(x), negative_slope=0.01)
        x = self.pool(x)
        x = F.leaky_relu(input=self.conv3(x), negative_slope=0.01)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        return x
    
input_linear_size = 288