# imports
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.optim import lr_scheduler  # Para fazer o fine Tunning gradual (variando o learning rate)
from sklearn.model_selection import train_test_split
from data_modules.pretext import HarDataset
from models.backbone import Backbone, input_linear_size
from utils.functions import saveBestModel, loadBestModel
from transforms.time_series import flip, negation, noise_addition, permutation, scalling

# hyperparameters
num_epoch = 1000
batch_size = 32
learning_rate = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
printQtd = 1

# dataloader
transforms = [flip, negation, noise_addition, permutation, scalling]

used_percent = 0.15
dataset = HarDataset(path="train")
train_index, _ = train_test_split(range(len(dataset)), test_size=1 - used_percent, shuffle=True)
subdataset = Subset(dataset, train_index)
data_loader = DataLoader(dataset=subdataset, batch_size=batch_size, shuffle=True, num_workers=2)

dataset_test = HarDataset(path="test")
data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True, num_workers=2)

# loadbackbone
backbone = Backbone()
try:
    backbone.load_state_dict(loadBestModel(path='best_models/', file_name='backbone', device=device))
    backbone.requires_grad_(True) 
    # for param in backbone.parameters():
    #     param.requires_grad = True
except:
    print("Erro ao carregar modelo de backbone!")

# define modelo de pretexto
class Conv1DNet(nn.Module):
    def __init__(self, backbone):
        super(Conv1DNet, self).__init__()
        self.backbone = backbone
        self.linear1 = nn.Linear(input_linear_size, 6)
        self.linear2 = nn.Linear(320, 6)
    
    def forward(self, x):
        x = self.backbone(x)
        x = F.leaky_relu(input=self.linear1(x), negative_slope=0.01)
        # x = F.leaky_relu(input=self.linear2(x), negative_slope=0.01)
        return x

model = Conv1DNet(backbone=backbone)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
n_total_steps = len(data_loader)
step_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.3)      # a cada 'step_size' epocas a learning rate é multiplicada por 'gamma'

# training loop
for epoch in range(num_epoch):
    for i, (data, labels) in enumerate(data_loader):
        outputs = model(data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % math.floor(n_total_steps/printQtd) == 0:
            print (f'{list(data.shape)} -> {list(labels.shape)} : Epoch [{epoch+1:4d}/{num_epoch}], Step [{i+1:4d}/{n_total_steps}], Loss: {loss.item():.4f}')

nums_labels = len(transforms) if len(dataset.transforms) > 0 else 6
accTotal = 0
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(nums_labels)]
    n_class_samples = [0 for i in range(nums_labels)]
    n_each_class_samples = [0 for i in range(nums_labels)]

    for data, labels in data_loader_test:
        outputs = model(data)

        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(labels.shape[0]):
            label = labels[i]
            pred  = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
            n_each_class_samples[pred] += 1

    accTotal = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {accTotal} %')

    for i in range(nums_labels):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {dataset_test.getLabel(i)} ({n_class_correct[i]}/{n_class_samples[i]} | {n_each_class_samples[i]}): {acc} %')


saveBestModel(accuracy=accTotal, batch_size=batch_size, epoch=num_epoch, model=model, path="best_models/", file_name="model")