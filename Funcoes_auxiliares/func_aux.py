import pandas as pd
import os
import unicodedata
import re

def func_apend_data(path: str, column=None):
    arquivos = os.listdir(path)
    lista = []
    for arq in arquivos:
        caminho = f'{path}\\{arq}'
        print(caminho)
        df = pd.read_csv(caminho, low_memory=False)
        if column is not None:
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

def func_categoriza_var_cnes(qtd: float):
    if qtd == 0:
        return 'nenhuma'
    elif 1 <= qtd <= 20:
        return 'entre_1_e_20'
    elif 21 <= qtd <= 40:
        return 'entre_21_e_40'
    elif 41 <= qtd <= 60:
        return 'entre_41_e_60'
    else:
        return 'mais_que_60'


# Dados da tabela female
data_female_10 = {
    14: 77, 15: 97, 16: 122, 17: 152, 18: 188, 19: 231, 20: 281, 21: 339, 22: 405, 23: 481,
    24: 567, 25: 663, 26: 769, 27: 886, 28: 1013, 29: 1150, 30: 1296, 31: 1451, 32: 1614, 33: 1783,
    34: 1957, 35: 2135, 36: 2314, 37: 2493, 38: 2670, 39: 2843, 40: 3010
}

data_female_90 = {
    14: 102, 15: 129, 16: 162, 17: 202, 18: 248, 19: 304, 20: 369, 21: 444, 22: 530, 23: 629,
    24: 740, 25: 865, 26: 1003, 27: 1156, 28: 1323, 29: 1505, 30: 1699, 31: 1907, 32: 2127, 33: 2358,
    34: 2598, 35: 2846, 36: 3099, 37: 3357, 38: 3616, 39: 3875, 40: 4131
}

# Dados da tabela male
data_male_10 = {
    14: 79, 15: 100, 16: 127, 17: 158, 18: 196, 19: 241, 20: 293, 21: 354, 22: 424, 23: 503,
    24: 592, 25: 692, 26: 803, 27: 924, 28: 1055, 29: 1197, 30: 1349, 31: 1509, 32: 1677, 33: 1852,
    34: 2032, 35: 2217, 36: 2404, 37: 2591, 38: 2778, 39: 2962, 40: 3142
}

data_male_90 = {
    14: 105, 15: 134, 16: 169, 17: 210, 18: 260, 19: 320, 20: 389, 21: 469, 22: 561, 23: 666,
    24: 785, 25: 917, 26: 1063, 27: 1224, 28: 1399, 29: 1587, 30: 1788, 31: 2000, 32: 2224, 33: 2456,
    34: 2694, 35: 2938, 36: 3185, 37: 3432, 38: 3676, 39: 3916, 40: 4149
}

def func_peso_calculado(sexo: int, peso: float, semana_gest: int):
    """
    :param sexo: 1 male; 2 female
    :param peso: fetal weight (g)
    :param semana_gest: gestational age in weeks
    :return: PIG, AIG or GIG
    """
    # Ajustar a semana gestacional para o intervalo válido
    semana_gest = max(14, min(40, semana_gest))

    # Selecionar os pesos de referência com base no sexo e na semana gestacional
    if sexo == 1:
        peso_10 = data_male_10[semana_gest]
        peso_90 = data_male_90[semana_gest]
    elif sexo == 2:
        peso_10 = data_female_10[semana_gest]
        peso_90 = data_female_90[semana_gest]
    else:
        return 'ignorado'

    # Determinar a categoria de peso
    if peso < peso_10:
        return 'PIG'
    elif peso_10 <= peso <= peso_90:
        return 'AIG'
    else:
        return 'GIG'


