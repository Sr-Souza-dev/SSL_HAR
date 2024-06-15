import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from converting.utils.enums import Sets

# A funcionalidade deste arquivo era extrair os valores de 59 instantes
# passados de cada base de dados, porém, os dados já pré-processados
# foram salvos e estão em download automático. Portanto, não é mais necessaria
# a utilização deste arquivo

def create_time_windows(df: pd.Series, n:np.integer, name='data'):
    df_copy = pd.DataFrame(data=df.values, columns=[name])
    
    for i in range(1, n+1):
        df_copy[f'{name}-{i}'] = df_copy[name].shift(periods=i)
    
    return df_copy

def get_each_col(data:pd.Series):
    print(data)

# Espera receber no columns o idx da col [accx, accy, accz, gyrx, gyry, gyrz]
def make_dataset(load_file, file, set=Sets.TRAIN.value, testSize = 0.2, split = True, columns = [], columnsNames = ['accx', 'accy', 'accz', 'gyrx', 'gyry', 'gyrz'], sep = " "):
    data = pd.read_csv(load_file, sep=sep, header= None, skiprows=1)
    print("Data Shape: ", data.shape)
    instante = 59
    new_data = pd.DataFrame({})
    for value, name in zip(columns, columnsNames):
        col = data.iloc[:, value]
        col = create_time_windows(df=col, n=instante, name=name)
        new_data = pd.concat([new_data, col], axis=1)
        print(f"{name}_shape: ",col.shape)

    new_data = new_data.dropna()
    print("dataframe_shape: ", new_data.shape)
    if split:
        # Salva o arquivo concatenado
        train_df, test_df = train_test_split(new_data, test_size=testSize, random_state=42)

        train_df.to_csv(f'{file}_{Sets.TRAIN.value}.dat', sep=' ', index=False, header=True)
        test_df.to_csv(f'{file}_{Sets.TEST.value}.dat', sep=' ', index=False, header=True)

        print("Train: ", train_df.shape)
        print("Teste: ",test_df.shape)
    else:
        # Salva o arquivo concatenado
        new_data.to_csv(f'{file}_{set}.dat', sep=' ', index=False, header=True)
        print(f"{file}: ", new_data.shape)

# Gera PAMAP2
# path = f"PAMAP2/subject101.dat"
# make_dataset(load_file=path, file = Datas.PAMAP.value, columns=[7, 8, 9, 10, 11, 12])

# Gera Mobit
# columns = [0, 1, 2, 120, 121, 122]
# make_dataset(sep=',', load_file="MOBIT/train.csv", file=Datas.MOBIT.value, set=Sets.TRAIN.value, columns=columns, split=False)
# make_dataset(sep=',', load_file="MOBIT/test.csv", file=Datas.MOBIT.value, set=Sets.TEST.value, columns=columns, split=False)

# Gera Motion
# columns = [11, 12, 13, 8, 9, 10]
# make_dataset(load_file="MOTION/data.dat", file=Datas.MOTION.value, columns=columns, testSize=0.3)

