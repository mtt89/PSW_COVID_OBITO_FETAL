import pandas as pd
import os

def func_apend_data(path: str, column: list):
    arquivos = os.listdir(path)
    lista = []
    for arq in arquivos:
        caminho = f'{path}\\{arq}'
        print(caminho)
        df = pd.read_csv(caminho, low_memory=False)
        df = df[column]
        lista.append(df)
    saida = pd.concat(lista)
    return saida

