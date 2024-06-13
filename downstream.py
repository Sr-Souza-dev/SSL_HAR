# imports
import torch
import math
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler  # Para fazer o fine Tunning gradual (variando o learning rate)
from data_modules.downstream import HarDataset
from models.backbone import Backbone, input_linear_size
from data.datas import Datas
from utils.functions import saveBestModel, loadBestModel


# hyperparameters
num_epoch = 500
batch_size = 10
learning_rate = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
printQtd = 1

# dataloader
print("Carregando base de dados...")
dataset_train = HarDataset(path="train")
data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)

dataset_validation = HarDataset(path="validation")
data_loader_validation = DataLoader(dataset=dataset_validation, batch_size=batch_size, shuffle=True, num_workers=2)

dataset_test = HarDataset(path="test")
data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True, num_workers=2)

nums_labels = len(dataset_train.labels)

# loadbackbone
print("Criando Modelo...")
backbone = Backbone()
try:
    backbone.load_state_dict(loadBestModel(path='best_models/', file_name=f'backbone_{Datas.MOTION.value}', device=device))
    backbone.requires_grad_(False) 
    # for param in backbone.parameters():
    #     param.requires_grad = True
except:
    print("Erro ao carregar modelo de backbone!")

# define modelo de pretexto
class Conv1DNet(nn.Module):
    def __init__(self, backbone):
        super(Conv1DNet, self).__init__()
        self.backbone = backbone
        self.linear1 = nn.Linear(input_linear_size, 320)
        self.linear2 = nn.Linear(320, nums_labels)
    
    def forward(self, x):
        x = self.backbone(x)
        x = F.leaky_relu(input=self.linear1(x), negative_slope=0.01)
        x = F.leaky_relu(input=self.linear2(x), negative_slope=0.01)
        return x

model = Conv1DNet(backbone=backbone)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
n_total_steps = len(data_loader_train)
step_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.5)      # a cada 'step_size' epocas a learning rate é multiplicada por 'gamma'

# training loop
print("Iniciando treinamento...")

train_errors = np.asarray([])
validation_errors = np.asarray([])
best_val_loss = 500
for epoch in range(num_epoch):
    # treinamento 
    model.train()
    train_loss = 0
    for i, (data, labels) in enumerate(data_loader_train):
        outputs = model(data)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % math.floor(n_total_steps/printQtd) == 0:
            print (f'Epoch [{epoch+1:4d}/{num_epoch}], Step [{i+1:4d}/{n_total_steps}], Loss: {loss.item():.4f}', end= "" if n_total_steps/printQtd+i >= n_total_steps else "\n")

    # Validação
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in data_loader_validation:
            output = model(data)
            val_loss += criterion(output, target).item()
    
    val_loss /= len(data_loader_validation)
    print(f' Validation Loss: {val_loss:.4f}')

    train_errors = np.append(train_errors, train_loss/len(data_loader_train))
    validation_errors = np.append(validation_errors, val_loss)

    if val_loss < best_val_loss:
        step_lr_scheduler.step()
        best_val_loss = val_loss
        best_model = model.state_dict()  # Salva os parâmetros do modelo


model.load_state_dict(best_model)

print("Iniciando avaliação...")
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


saveBestModel(accuracy=accTotal, batch_size=batch_size, epoch=num_epoch, model=model, path="best_models/", file_name="downstream")

df = pd.DataFrame({
    "train" : train_errors,
    "validation" : validation_errors
})
df.to_csv("results_data/downstream_train.dat", sep=" ", index=False)