import torch
from pathlib import Path
import os

def verifyFile(path):
    return os.path.isfile(path)

def verifyPath(path):
    folder = Path(path)
    if not folder.exists():
        folder.mkdir()

def saveBestModel(path, batch_size, epoch, accuracy, model: torch.nn.Module, file_name='model'):
    checkpoint = {
        "batch_size": batch_size,
        "epoch": epoch,
        "accuracy": accuracy,
        "model": model.state_dict()
    }

    fullpath = f"{path}{file_name}.pth"
    verifyPath(path)
    last_accuracy = 0
    if verifyFile(fullpath):
        last_checkpoint = torch.load(fullpath)
        last_accuracy = last_checkpoint['accuracy']

    if(last_accuracy < accuracy):
        torch.save(checkpoint, fullpath)

def loadBestModel(path, device, file_name='model'):
    fullpath = f"{path}{file_name}.pth"
    if verifyFile(fullpath):
        last_checkpoint = torch.load(fullpath, map_location=device)
        return last_checkpoint['model']
    print(f"Modelo '{file_name}' não encontrado!")

def loadCheckPointData(path, device, file_name='model'):
    fullpath = f"{path}{file_name}.pth"
    if verifyFile(fullpath):
        last_checkpoint = torch.load(fullpath, map_location=device)
        return last_checkpoint
    print(f"Modelo '{file_name}' não encontrado!")

def printModelWeights(model):
    # Print the first layer of the model
    for name, param in model.named_parameters():
        print(f"Nome do parâmetro: {name}, Valor: {param}")
        break