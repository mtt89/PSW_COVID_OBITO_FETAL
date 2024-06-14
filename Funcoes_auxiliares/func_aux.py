import pandas as pd
import os
import unicodedata
import re

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

def func_categorize_peso(peso):
    if peso < 500:
        return "menor_500"
    elif 500 <= peso <= 1499:
        return "entre_500_1499"
    elif 1500 <= peso <= 2499:
        return "entre_1500_2499"
    elif 2500 <= peso <= 3500:
        return "entre_2500_3500"
    elif 3500 <= peso <= 3999:
        return "entre_3500_3999"
    elif peso >= 4000:
        return "maior_igual_4000"
    else:
        return "Ignorado"
    
def func_categorize_idademae(idade):
    if idade <= 19:
        return "menor_igual_19"
    elif 20 <= idade <= 34:
        return "entre_20_34"
    elif 35 <= idade <= 39:
        return "entre_35_39"
    elif idade >= 40:
        return "maior_igual_40"
    else:
        return "Ignorado"

def func_categorize_escolmae(cod_escol):
    if cod_escol == 0:
        return "Sem_escolaridade"
    elif 1 <= cod_escol <= 2:
        return "Fundamental"
    elif cod_escol == 3:
        return "Ensino_medio"
    elif 4 <= cod_escol <= 5:
        return "Ensino_superior"
    else:
        return "Ignorado"

def func_categorize_gravidez(cod_grav):
    if cod_grav == 1:
        return "Unica"
    elif 2 <= cod_grav <= 3:
        return "Multipla"
    else:
        return "Ignorado"
    
def func_categorize_idade_gest(semana):
    if semana < 22:
        return "menor_22"
    elif 22 <= semana <= 27:
        return "entre_22_27"
    elif 28 <= semana <= 36:
        return "entre_28_36"
    elif 37 <= semana <= 39:
        return "entre_37_39"
    elif 40 <= semana <= 42:
        return "entre_40_42"
    else:
        return "Ignorado"

def func_limpar_string(texto):
    # Remove accentuation
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    # Removes special characters and punctuation except the dash
    texto = re.sub(r'[^a-zA-Z0-9\s-]', '', texto)
    # Convert to uppercase
    texto = texto.upper()
    # Remove extra whitespace
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto