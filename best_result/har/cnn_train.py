import sys
sys.path.append('../../')
from utils.enums import Datas, main_data, teste_size
from utils.checkpoints import verifyPath

path_reports = f"../../report_results/{Datas.HAR.value}/{main_data.value}_{teste_size}/"

split_path = path_reports.split("/")
partial_path = ""
for i, part in enumerate(split_path):
    partial_path += part + "/"
    verifyPath(partial_path)

import torch
import math
import pandas as pd
from data_modules.har import HarDataModule as HarDataModuleDownstram
from utils.enums import Datas, Sets, ModelTypes
from models.cnn1d import CNN1d

printQtd = 1
batch_size = 5
num_epoch = 200
isFreezing = False
with_validation = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Optim config backbone
learning_rate_bb = 0.02
step_size_bb = 100
gamma_bb = 0.5

# Optim config pred_head
learning_rate_ds = 0.01
step_size_ds = 100
gamma_ds = 0.5

# Optim multi test configs
learning_rate = 0.01
step_size = 200
gamma = 0.5

data_module = HarDataModuleDownstram(batch_size=batch_size, ref="../../")
train_dl, train_ds = data_module.get_dataloader(set=Sets.TRAIN.value, shuffle=True)
test_dl, test_ds   = data_module.get_dataloader(set=Sets.TEST.value, shuffle=True)
num_classes = len(train_ds.labels)

model = CNN1d(
    data_label=main_data.value, 
    num_classes=num_classes, 
    require_grad= not isFreezing, 
    type=ModelTypes.DOWNSTREAM.value,
    ref="../../"
)
model.require_grads_backbone(not isFreezing)
print(model)

# import lightning as L
# trainer = L.Trainer(
#     max_epochs=10,
#     accelerator='cpu',
#     log_every_n_steps=1        
# )
# trainer.fit(model=model, train_dataloaders=train_dl)

optimizer_backbone, lr_scheduler_backbone = model.configure_backbone_optimizers(step_size=step_size_bb, gamma=gamma_bb, learning_rate=learning_rate_bb)
optimizer_downstream, lr_scheduler_downstream = model.configure_head_optimizers(step_size=step_size_ds, gamma=gamma_ds, learning_rate=learning_rate_ds)

train_errors = []
validation_errors = []
best_val_loss = 500
n_total_steps = len(train_dl)
for epoch in range(num_epoch):
    # Treinamento
    model.train()
    train_loss = 0
    for i, batch in enumerate(train_dl):
        loss = model.training_step(batch)
        train_loss += loss.item()
        optimizer_backbone.zero_grad()
        optimizer_downstream.zero_grad()
        loss.backward()
        optimizer_backbone.step()
        optimizer_downstream.step()
        
        if (i+1) % math.floor(n_total_steps/printQtd) == 0:
            print (f'Epoch [{epoch+1:4d}/{num_epoch}], Step [{i+1:4d}/{n_total_steps}], Loss: {loss.item():.4f}', end= "" if n_total_steps/printQtd+i >= n_total_steps and with_validation else "\n")

    lr_scheduler_backbone.step()
    lr_scheduler_downstream.step()

    if with_validation:  
        # Validação
        model.eval() 
        val_loss = 0
        with torch.no_grad():
            for batch in test_dl:
                val_loss += model.validation_step(batch)
        
        val_loss /= len(test_dl)
        print(f' Validation Loss: {val_loss:.4f}')

        train_errors.append(train_loss/len(train_dl))
        validation_errors.append(val_loss.item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()  # Salva os parâmetros do modelo
    else:
        best_model = model.state_dict()


model.load_state_dict(best_model)

accTotal = 0
predicted_values = []
real_values = []
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(num_classes)]
    n_class_samples = [0 for i in range(num_classes)]
    n_each_class_samples = [0 for i in range(num_classes)]

    for data, labels in test_dl:
        outputs = model(data)

        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for pred, real in zip (predicted, labels):
            predicted_values.append(pred.item())
            real_values.append(real.item())

        for i in range(labels.shape[0]):
            label = labels[i]
            pred  = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
            n_each_class_samples[pred] += 1

    accTotal = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {accTotal} %')

    for i in range(num_classes):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {test_ds.getLabel(i)} ({n_class_correct[i]}/{n_class_samples[i]} | {n_each_class_samples[i]}): {acc} %')

model.save_full_model(accuracy=accTotal, batch_size=batch_size, num_epoch=num_epoch)

pred_reports = pd.DataFrame({
    Sets.REAL.value: real_values,
    Sets.PREDICTION.value : predicted_values
})
pred_reports.to_csv(f"{path_reports}/predictions_{ModelTypes.DOWNSTREAM.value}.dat", sep=" ", index=False)

train_reports = pd.DataFrame({
    Sets.TRAIN.value : train_errors,
    Sets.VALIDATION.value : validation_errors
})
train_reports.to_csv(f"{path_reports}/errors_{ModelTypes.DOWNSTREAM.value}.dat", sep=" ", index=False)

percents = [1, 20, 40, 60, 80, 100]

for percent in percents:

    # Carrega a base de dados
    data_module = HarDataModuleDownstram(batch_size=batch_size, ref="../../")
    train_dl, train_ds = data_module.get_dataloader(set=Sets.TRAIN.value, shuffle=True, percent=percent/100)
    test_dl, test_ds   = data_module.get_dataloader(set=Sets.TEST.value, shuffle=True)
    num_classes = len(test_ds.labels)

    # Define os modelos
    model1 = CNN1d(
        data_label=main_data.value, 
        num_classes=num_classes, 
        require_grad= not isFreezing, 
        type=ModelTypes.DOWNSTREAM.value,
        ref="../../"
    )
    model2 = CNN1d(
        data_label=main_data.value, 
        num_classes=num_classes, 
        require_grad=True, 
        type=ModelTypes.PRETEXT.value
    )

    # Define os otimizadores
    optimizer1, lr_scheduler1 = model1.configure_optimizers(step_size=step_size, gamma=gamma, learning_rate=learning_rate)
    optimizer2, lr_scheduler2 = model2.configure_optimizers(step_size=step_size, gamma=gamma, learning_rate=learning_rate)

    optimizer1, lr_scheduler1 = optimizer1[0], lr_scheduler1[0]
    optimizer2, lr_scheduler2 = optimizer2[0], lr_scheduler2[0]

    # Realiza o treinamento
    best_val_loss1 = 500
    best_val_loss2 = 500
    n_total_steps = len(train_dl)
    for epoch in range(num_epoch):
        
        # Treinamento 
        model1.train()
        model2.train()
        for i, batch in enumerate(train_dl):
            if len(batch) <= 0:
                break
            loss1 = model1.training_step(batch)
            loss2 = model2.training_step(batch)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss1.backward()
            loss2.backward()

            optimizer1.step()
            optimizer2.step()

        lr_scheduler1.step()
        lr_scheduler2.step()

        if with_validation:
            # Validação
            model1.eval()
            model2.eval()

            val_loss1 = 0
            val_loss2 = 0

            with torch.no_grad():
                for batch in test_dl:
                    val_loss1 += model1.validation_step(batch)
                    val_loss2 += model2.validation_step(batch)
            
            val_loss1 /= len(test_dl)
            val_loss2 /= len(test_dl)

            if val_loss1 < best_val_loss1:
                best_val_loss1 = val_loss1
                best_model1 = model1.state_dict() 

            if val_loss2 < best_val_loss2:
                best_val_loss2 = val_loss2
                best_model2 = model2.state_dict()  
        else:
            best_model1 = model1.state_dict() 
            best_model2 = model2.state_dict()  

    model1.load_state_dict(best_model1)
    model2.load_state_dict(best_model2)

    predicted_values1 = []
    predicted_values2 = []

    real_values1 = []
    real_values2 = []
    
    with torch.no_grad():
        for data, labels in test_dl:
            outputs1 = model1(data)
            _, predicted1 = torch.max(outputs1, 1)

            for pred, real in zip (predicted1, labels):
                predicted_values1.append(pred.item())
                real_values1.append(real.item())

            outputs2 = model2(data)
            _, predicted2 = torch.max(outputs2, 1)

            for pred, real in zip (predicted2, labels):
                predicted_values2.append(pred.item())
                real_values2.append(real.item())

    pred_reports1 = pd.DataFrame({
        Sets.REAL.value: real_values1,
        Sets.PREDICTION.value : predicted_values1
    })
    pred_reports2 = pd.DataFrame({
        Sets.REAL.value: real_values2,
        Sets.PREDICTION.value : predicted_values2
    })

    pred_reports1.to_csv(f"../../report_results/har/percent/cnn/predictions_{percent}_m1.dat", sep=" ", index=False)
    pred_reports2.to_csv(f"../../report_results/har/percent/cnn/predictions_{percent}_m2.dat", sep=" ", index=False)