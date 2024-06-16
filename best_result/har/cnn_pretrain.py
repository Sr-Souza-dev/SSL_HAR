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

import math
import torch
import pandas as pd
from models.cnn1d import CNN1d
from utils.enums import Datas, Sets, ModelTypes
from data_modules.pretext import HarDataModule as HarDataModulePretext
from transforms.har import rotation, flip, noise_addition, permutation, scaling, time_warp, negation

printQtd = 1
num_epoch = 320
batch_size = 32
with_validation = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Optim
learning_rate = 0.02
step_size = 80
gamma = 0.5

transforms = [rotation, flip, noise_addition, permutation, scaling, time_warp, negation]
data_module = HarDataModulePretext(batch_size=batch_size, main_data = main_data, ref="../../")
train_dl, train_ds = data_module.get_dataloader(set=Sets.TRAIN.value, shuffle=True, transforms=transforms)
test_dl, test_ds   = data_module.get_dataloader(set=Sets.TEST.value, shuffle=False, transforms=transforms)
num_classes = len(transforms)

model_p = CNN1d(data_label=main_data.value, num_classes=num_classes, require_grad=True, type=ModelTypes.PRETEXT.value, ref="../../")
print(model_p)

# import lightning as L
# trainer = L.Trainer(
#     max_epochs=10,
#     accelerator='cpu',
#     log_every_n_steps=1        
# )
# trainer.fit(model=model, train_dataloaders=train_dl)

optimizer, lr_scheduler = model_p.configure_optimizers(step_size=step_size, gamma=gamma, learning_rate=learning_rate)
optimizer, lr_scheduler = optimizer[0], lr_scheduler[0]

train_errors = []
validation_errors = []
best_val_loss = 500
n_total_steps = len(train_dl)
for epoch in range(num_epoch):
    # Treinamento
    model_p.train()
    train_loss = 0
    for i, batch in enumerate(train_dl):
        loss = model_p.training_step(batch)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % math.floor(n_total_steps/printQtd) == 0:
            print (f'Epoch [{epoch+1:4d}/{num_epoch}], Step [{i+1:4d}/{n_total_steps}], Loss: {loss.item():.4f}', end= "" if n_total_steps/printQtd+i >= n_total_steps and with_validation else "\n")

    lr_scheduler.step()

    if with_validation:
        # Validação
        model_p.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_dl:
                val_loss += model_p.validation_step(batch)
        
        val_loss /= len(test_dl)
        print(f' Validation Loss: {val_loss:.4f}')

        train_errors.append(train_loss/len(train_dl))
        validation_errors.append(val_loss.item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model_p.state_dict()  # Salva os parâmetros do modelo
    else:
        best_model = model_p.state_dict()

model_p.load_state_dict(best_model)

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
        outputs = model_p(data)

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


model_p.save_backbone(accuracy=accTotal, batch_size=batch_size, num_epoch=num_epoch)

pred_reports = pd.DataFrame({
    Sets.REAL.value: real_values,
    Sets.PREDICTION.value : predicted_values
})
pred_reports.to_csv(f"{path_reports}/predictions_{ModelTypes.PRETEXT.value}.dat", sep=" ", index=False)

train_reports = pd.DataFrame({
    Sets.TRAIN.value : train_errors,
    Sets.VALIDATION.value : validation_errors
})
train_reports.to_csv(f"{path_reports}/errors_{ModelTypes.PRETEXT.value}.dat", sep=" ", index=False)