import pandas as pd
import numpy as np
import random
import os

# A funcionalidade deste arquivo era para extrair valores de forma aleatória da base de dados
# original do motion activity.

def list_folders(caminho_base):
    # Lista para armazenar os nomes dos diretórios
    diretorios = []
    
    # Itera sobre os itens no diretório base
    for item in os.listdir(caminho_base):
        item_caminho = os.path.join(caminho_base, item)
        
        # Verifica se o item é um diretório
        if os.path.isdir(item_caminho):
            diretorios.append(item)
    
    return diretorios

def list_files(caminho_base):
    # Lista para armazenar os nomes dos arquivos
    arquivos = []
    
    # Itera sobre os itens no diretório base
    for item in os.listdir(caminho_base):
        item_caminho = os.path.join(caminho_base, item)
        
        # Verifica se o item é um arquivo
        if os.path.isfile(item_caminho) and item.endswith('.csv'):
            arquivos.append(item)
    
    return arquivos

max_count = 3
get_qtd_each_file = 100


folders = sorted(list_folders('./'))
data = pd.DataFrame({})
for fold in folders:
    files = list_files(fold+"/")
    random.shuffle(files)
    count = 0
    for file in files:
        if max_count < count:
            break
        count += 1
        newData = pd.read_csv(fold+"/"+file, sep=',')
        randomValue = random.randint(0, len(newData)-get_qtd_each_file-5)
        newData = newData.iloc[randomValue:randomValue+get_qtd_each_file,:]
        data = pd.concat([data, newData], axis=0)

print(data)
data.to_csv("./data.dat", sep=" ")
