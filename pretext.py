# imports
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_modules.pretext import PamapDataset
from models.backbone import Backbone, input_linear_size
from utils.functions import saveBestModel
from transforms.time_series import rotation, flip, noise_addition, permutation, scaling, time_warp, negation

# hyperparameters
num_epoch = 200
batch_size = 1024
learning_rate = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
printQtd = 1

# dataloader
print("Carregando base de dados...")
transforms = [rotation, flip, noise_addition, permutation, scaling, time_warp, negation]
dataset = PamapDataset(set="train", transforms = transforms)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
nums_labels = len(transforms)

dataset_test = PamapDataset(set="test", transforms = transforms)
data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True, num_workers=2)

# models
print("Criando Modelo...")
class Conv1DNet(nn.Module):
    def __init__(self, backbone):
        super(Conv1DNet, self).__init__()
        self.backbone = backbone
        self.linear1 = nn.Linear(input_linear_size, nums_labels)
        self.linear2 = nn.Linear(320, 6)
    
    def forward(self, x):
        x = self.backbone(x)
        x = F.leaky_relu(input=self.linear1(x), negative_slope=0.01)
        # x = F.leaky_relu(input=self.linear2(x), negative_slope=0.01)
        return x

backbone = Backbone()
model = Conv1DNet(backbone=backbone)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
n_total_steps = len(data_loader)

# training loop
print("Iniciando treinamento...")
for epoch in range(num_epoch):
    # Treinamento
    model.train()
    for i, (data, labels) in enumerate(data_loader):
        outputs = model(data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % math.floor(n_total_steps/printQtd) == 0:
           print (f'Epoch [{epoch+1:4d}/{num_epoch}], Step [{i+1:4d}/{n_total_steps}], Loss: {loss.item():.4f}', end= "" if n_total_steps/printQtd+i >= n_total_steps else "\n")
    
    # Validação
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in data_loader_test:
            output = model(data)
            val_loss += criterion(output, target).item()
    
    val_loss /= len(data_loader_test)
    print(f' Validation Loss: {val_loss:.4f}')



accTotal = 0
print("Iniciando avaliação...")
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

backbone = backbone
saveBestModel(accuracy=accTotal, batch_size=batch_size, epoch=num_epoch, model=backbone, path="best_models/", file_name="backbone")