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
data_female = {
    14: 77, 15: 97, 16: 122, 17: 152, 18: 188, 19: 231, 20: 281, 21: 339, 22: 405, 23: 481,
    24: 567, 25: 663, 26: 769, 27: 886, 28: 1013, 29: 1150, 30: 1296, 31: 1451, 32: 1614, 33: 1783,
    34: 1957, 35: 2135, 36: 2314, 37: 2493, 38: 2670, 39: 2843, 40: 3010
}

# Dados da tabela male
data_male = {
    14: 79, 15: 100, 16: 127, 17: 158, 18: 196, 19: 241, 20: 293, 21: 354, 22: 424, 23: 503,
    24: 592, 25: 692, 26: 803, 27: 924, 28: 1055, 29: 1197, 30: 1349, 31: 1509, 32: 1677, 33: 1852,
    34: 2032, 35: 2217, 36: 2404, 37: 2591, 38: 2778, 39: 2962, 40: 3142
}

def func_peso_calculado(sexo: int, peso: float, semana_gest: int):
    """
    :param sexo: 1 male; 2 female
    :param peso: fetal weight (g)
    :param semana_gest: gestational age in weeks
    :return: pequeno < peso do estudo, adequado = peso do estudo, gig > peso do estudo
    """
    # Ajustar a semana gestacional para o intervalo válido
    semana_gest = max(14, min(40, semana_gest))

    # Selecionar o peso de referência com base no sexo e na semana gestacional
    if sexo == 1:
        peso_ref = data_male[semana_gest]
    elif sexo == 2:
        peso_ref = data_female[semana_gest]
    else:
        return 'ignorado'

    # Determinar a categoria de peso
    if peso > peso_ref:
        return 'gig'
    elif peso < peso_ref:
        return 'pequeno'
    else:
        return 'adequado'


# def func_peso_calculado(sexo: int, peso: float, semana_gest: int):
#     """
#
#     :param sexo: 1 male; 2 female
#     :param peso: fetal weight (g)
#     :param semana_gest: gestational age in weeks
#     :return: pequeno < peso do estudo, adequado = peso do estudo, gig > peso do estudo
#     """
#     # Dados da tabela female
#     data_female = {
#         'semana': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
#                    39, 40],
#         'female_5': [73, 92, 116, 145, 180, 221, 269, 324, 388, 461, 542, 634, 735, 846, 967, 1096, 1234, 1379, 1530,
#                      1687, 1847, 2008, 2169, 2329, 2484, 2633, 2775],
#         'female_10': [77, 97, 122, 152, 188, 231, 281, 339, 405, 481, 567, 663, 769, 886, 1013, 1150, 1296, 1451, 1614,
#                       1783, 1957, 2135, 2314, 2493, 2670, 2843, 3010],
#         'female_25': [82, 104, 131, 164, 202, 248, 302, 364, 435, 516, 608, 710, 823, 948, 1083, 1230, 1386, 1553, 1728,
#                       1911, 2101, 2296, 2494, 2695, 2896, 3096, 3294],
#         'female_50': [89, 113, 141, 176, 217, 266, 322, 388, 464, 551, 649, 758, 880, 1014, 1160, 1319, 1489, 1670,
#                       1861, 2060, 2268, 2481, 2698, 2917, 3136, 3354, 3567],
#         'female_75': [96, 121, 152, 189, 233, 285, 346, 417, 499, 592, 697, 815, 946, 1090, 1247, 1418, 1601, 1796,
#                       2002, 2217, 2440, 2669, 2902, 3138, 3373, 3605, 3832],
#         'female_90': [102, 129, 162, 202, 248, 304, 369, 444, 530, 629, 740, 865, 1003, 1156, 1323, 1505, 1699, 1907,
#                       2127, 2358, 2598, 2846, 3099, 3357, 3616, 3875, 4131],
#         'female_95': [107, 135, 170, 211, 261, 319, 387, 466, 557, 660, 776, 907, 1051, 1210, 1383, 1570, 1770, 1984,
#                       2209, 2445, 2690, 2943, 3201, 3462, 3725, 3988, 4247]
#     }
#
#     # Dados da tabela male
#     data_male = {
#         'semana': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
#                    39, 40],
#         'male_5': [75, 96, 121, 152, 188, 232, 282, 341, 408, 484, 570, 666, 772, 888, 1014, 1149, 1293, 1445, 1605,
#                    1770, 1941, 2114, 2290, 2466, 2641, 2813, 2981],
#         'male_10': [79, 100, 127, 158, 196, 241, 293, 354, 424, 503, 592, 692, 803, 924, 1055, 1197, 1349, 1509, 1677,
#                     1852, 2032, 2217, 2404, 2591, 2778, 2962, 3142],
#         'male_25': [84, 107, 136, 170, 210, 258, 314, 380, 454, 539, 635, 742, 860, 989, 1129, 1281, 1442, 1613, 1793,
#                     1980, 2174, 2372, 2574, 2777, 2981, 3183, 3382],
#         'male_50': [92, 116, 146, 183, 226, 277, 337, 407, 487, 578, 681, 795, 923, 1063, 1215, 1379, 1555, 1741, 1937,
#                     2140, 2350, 2565, 2783, 3001, 3218, 3432, 3639],
#         'male_75': [99, 126, 158, 197, 243, 298, 362, 436, 522, 619, 730, 853, 990, 1141, 1305, 1482, 1672, 1874, 2085,
#                     2306, 2534, 2767, 3002, 3238, 3472, 3701, 3923],
#         'male_90': [105, 134, 169, 210, 260, 320, 389, 469, 561, 666, 785, 917, 1063, 1224, 1399, 1587, 1788, 2000,
#                     2224, 2456, 2694, 2938, 3185, 3432, 3676, 3916, 4149],
#         'male_95': [109, 139, 175, 219, 271, 333, 405, 489, 586, 695, 818, 956, 1109, 1276, 1458, 1654, 1863, 2085,
#                     2319, 2562, 2814, 3072, 3334, 3598, 3863, 4125, 4383]
#     }
#     # Criar DataFrames
#     df_female = pd.DataFrame(data_female)
#     df_male = pd.DataFrame(data_male)
#
#     # acertando a semana
#     if semana_gest < 14:
#         semana_gest = 14
#     elif semana_gest > 40:
#         semana_gest = 40
#     else:
#         semana_gest = semana_gest
#
#     if sexo == 1:
#        peso_ref = df_male.loc[df_male['semana'] == semana_gest, ['male_10']].reset_index(drop=True)['male_10'][0]
#        if peso > peso_ref:
#            return 'gig'
#        elif peso < peso_ref:
#            return 'pequeno'
#        elif peso == peso_ref:
#            return 'adequado'
#        else:
#            return 'ignorado'
#     elif sexo == 2:
#        peso_ref = df_female.loc[df_female['semana'] == semana_gest, ['female_10']].reset_index(drop=True)['female_10'][0]
#        if peso > peso_ref:
#            return 'gig'
#        elif peso < peso_ref:
#            return 'pequeno'
#        elif peso == peso_ref:
#            return 'adequado'
#        else:
#            return 'ignorado'
#     else:
#        return 'ignorado'

