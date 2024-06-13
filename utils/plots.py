import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_instance_data(data: torch.Tensor, old_data: torch.Tensor, title = "transformation", oldTitle = "Orignal"):
    np_data = data.numpy()
    np_old_data = old_data.numpy()

    fig, axs = plt.subplots(2, 2, figsize=(14, 6))

    for i in range(np_data.shape[0]):
        if i > 2 :
            axs[1,0].plot(np_old_data[i], label=f'Curva {i+1}')
            axs[1,1].plot(np_data[i], label=f'Curva {i+1}')
        else:
            axs[0,0].plot(np_old_data[i], label=f'Curva {i+1}')
            axs[0,1].plot(np_data[i], label=f'Curva {i+1}')
    
    axs[0,0].set_title(f'{oldTitle}_Acc')
    axs[0,0].set_xlabel('Índice')
    axs[0,0].set_ylabel('Valor')
    axs[0,0].legend()
    axs[1,0].set_title(f'{oldTitle}_Acc')
    axs[1,0].set_xlabel('Índice')
    axs[1,0].set_ylabel('Valor')
    axs[1,0].legend()

    axs[0,1].set_title(f'{title}_Acc')
    axs[0,1].set_xlabel('Índice')
    axs[0,1].set_ylabel('Valor')
    axs[0,1].legend()
    axs[1,1].set_title(f'{title}_Gyr')
    axs[1,1].set_xlabel('Índice')
    axs[1,1].set_ylabel('Valor')
    axs[1,1].legend()

    # Ajustar layout para evitar sobreposição
    plt.tight_layout()

    # Mostrar o gráfico
    plt.show()

def plot_curves(data1: np.array, data2: np.array, title = "Plot Curves", data1_legend = "Data 1", data2_legend = "Data 2"):
    plt.plot(data1, label={data1_legend})
    plt.plot(data2, label={data2_legend})
    plt.title(title)
    plt.xlabel('Índice')
    plt.ylabel('Valor')
    plt.legend()
    plt.show()