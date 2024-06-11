import pandas as pd
from Funcoes_auxiliares.func_aux import func_apend_data

"""
--------------------------------- Due to the volume it may take some time ----------------------------------------------
from Download_data.import_data_pcdas import *
Import data
PCDAS = Import_Data_PCDAS()
path_= 'C:/Users/gabri/Documents/PSW_COVID_OBITO_FETAL'
year=[2019, 2020, 2021]
PCDAS.function_import_sim_dofet(path_download=path_, year=year)
PCDAS.function_import_sinasc(path_download=path_, year=year)
PCDAS.function_import_cnes(path_download=path_, year=year)

Obs: 
 Another option is to download the data manually, for this see the file 'manual_download_data.py'
------------------------------------------------------------------------------------------------------------------------
"""

#------------------------------------------------- SIM_DOFET -----------------------------------------------------------
caminho = 'C:/Users/gabri/Documents/PSW_COVID_OBITO_FETAL/SIM_DOFET'
variaveis = colunas= [
    'data_obito'
    , 'ano_obito'
    , 'TIPOBITO'
    , 'ocor_MUNNOMEX'
    , 'res_MUNNOMEX'
    , 'ocor_CAPITAL'
    , 'res_CAPITAL'
    , 'ocor_REGIAO'
    , 'res_REGIAO'
    , 'IDADEMAE'
    , 'ESCMAE'
    , 'def_escol_mae'
    , 'ESCMAE2010'
    , 'GRAVIDEZ'
    , 'def_gravidez'
    , 'GESTACAO'
    , 'def_gestacao'
    , 'SEMAGESTAC'
    , 'SEXO'
    , 'def_sexo'
    , 'PESO'
    , 'OBITOPARTO'
    , 'def_obito_parto'
    , 'CAUSABAS'
    , 'causabas_capitulo'
    , 'causabas_categoria'
    , 'causabas_grupo'
    , 'causabas_subcategoria'
    ]
df_sim_dofet = func_apend_data(path=caminho, column=)

