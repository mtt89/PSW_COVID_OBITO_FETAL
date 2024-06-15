import pandas as pd
from Funcoes_auxiliares.func_aux import *

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
variaveis = [
    'data_obito'
    , 'ano_obito'
    , 'TIPOBITO'
    , 'ocor_MUNNOMEX'
    , 'res_MUNNOMEX'
    , 'ocor_CAPITAL'
    , 'res_CAPITAL'
    , 'ocor_REGIAO'
    , 'res_REGIAO'
    , 'ocor_SIGLA_UF'
    , 'res_SIGLA_UF'
    , 'IDADEMAE'
    #, 'ESCMAE'
    #, 'def_escol_mae'
    , 'ESCMAE2010'
    , 'OBITOGRAV'
    , 'GRAVIDEZ'
    #, 'def_gravidez'
    #, 'GESTACAO'
    #, 'def_gestacao'
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

df_sim_dofet = func_apend_data(path=caminho, column=variaveis)
df_sim_dofet['FLAG_BASE'] = 'SIM_DOFET'
df_sim_dofet['idademae_faixa'] = [func_categorize_idademae(i) for i in df_sim_dofet['IDADEMAE']]
df_sim_dofet['escolaridade_mae'] = [func_categorize_escolmae(i) for i in df_sim_dofet['ESCMAE2010']]
df_sim_dofet['tipo_gravidez'] = [func_categorize_gravidez(i) for i in df_sim_dofet['GRAVIDEZ']]
df_sim_dofet['idade_gestacao_faixa'] = [func_categorize_idade_gest(i) for i in df_sim_dofet['SEMAGESTAC']]
df_sim_dofet['peso_faixa'] = [func_categorize_peso(i) for i in df_sim_dofet['PESO']]

df_sim_dofet = df_sim_dofet[
    [
    'data_obito'
    , 'ano_obito'
    , 'TIPOBITO'
    , 'ocor_MUNNOMEX'
    , 'res_MUNNOMEX'
    , 'ocor_CAPITAL'
    , 'res_CAPITAL'
    , 'ocor_REGIAO'
    , 'res_REGIAO'
    , 'ocor_SIGLA_UF'
    , 'res_SIGLA_UF'
    , 'IDADEMAE'
    , 'idademae_faixa'
    , 'ESCMAE2010'
    , 'escolaridade_mae'
    , 'OBITOGRAV'
    , 'GRAVIDEZ'
    , 'tipo_gravidez'
    , 'SEMAGESTAC'
    , 'idade_gestacao_faixa'
    , 'SEXO'
    , 'def_sexo'
    , 'PESO'
    , 'peso_faixa'
    , 'OBITOPARTO'
    , 'def_obito_parto'
    , 'CAUSABAS'
    , 'causabas_capitulo'
    , 'causabas_categoria'
    , 'causabas_grupo'
    , 'causabas_subcategoria'
    , 'FLAG_BASE'
    ]
]

df_sim_dofet.to_csv('base_suja/base_sim_dofet_suja.csv', index=False)

#------------------------------------------------ SINASC ---------------------------------------------------------------
caminho = 'C:/Users/gabri/Documents/PSW_COVID_OBITO_FETAL/SINASC'
variaveis = [
    'data_nasc'
    , 'ano_nasc'
   # , 'TIPOBITO' # Inserir
    , 'nasc_MUNNOMEX' #'ocor_MUNNOMEX'
    , 'res_MUNNOMEX'
    , 'nasc_CAPITAL'
    , 'res_CAPITAL'
    , 'nasc_REGIAO'
    , 'res_REGIAO'
    , 'IDADEMAE'
    #, 'ESCMAE'
    #, 'def_escol_mae'
    , 'ESCMAE2010'
    # , 'OBITOGRAV' Inserir
    , 'GRAVIDEZ'
    #, 'def_gravidez'
    #, 'GESTACAO'
    #, 'def_gestacao'
    , 'SEMAGESTAC'
    , 'SEXO'
    , 'def_sexo'
    , 'PESO'
    # , 'OBITOPARTO' Inserir
    # , 'def_obito_parto' Inserir
    # , 'CAUSABAS' Inserir
    # , 'causabas_capitulo' Inserir
    # , 'causabas_categoria' Inserir
    # , 'causabas_grupo' Inserir
    # , 'causabas_subcategoria' Inserir
    ]

df_sinasc = func_apend_data(path=caminho, column=variaveis)
df_sinasc['FLAG_BASE'] = 'SINASC'
df_sinasc['idademae_faixa'] = [func_categorize_idademae(i) for i in df_sinasc['IDADEMAE']]
df_sinasc['escolaridade_mae'] = [func_categorize_escolmae(i) for i in df_sinasc['ESCMAE2010']]
df_sinasc['tipo_gravidez'] = [func_categorize_gravidez(i) for i in df_sinasc['GRAVIDEZ']]
df_sinasc['idade_gestacao_faixa'] = [func_categorize_idade_gest(i) for i in df_sinasc['SEMAGESTAC']]
df_sinasc['peso_faixa'] = [func_categorize_peso(i) for i in df_sinasc['PESO']]

df_sinasc = df_sinasc[
    [
    'data_nasc'
    , 'ano_nasc'
    # , 'TIPOBITO'
    , 'nasc_MUNNOMEX'
    , 'res_MUNNOMEX'
    , 'nasc_CAPITAL'
    , 'res_CAPITAL'
    , 'nasc_REGIAO'
    , 'res_REGIAO'
    , 'IDADEMAE'
    , 'idademae_faixa'
    , 'ESCMAE2010'
    , 'escolaridade_mae'
    # , 'OBITOGRAV'
    , 'GRAVIDEZ'
    , 'tipo_gravidez'
    , 'SEMAGESTAC'
    , 'idade_gestacao_faixa'
    , 'SEXO'
    , 'def_sexo'
    , 'PESO'
    , 'peso_faixa'
    # , 'OBITOPARTO'
    # , 'def_obito_parto'
    # , 'CAUSABAS'
    # , 'causabas_capitulo'
    # , 'causabas_categoria'
    # , 'causabas_grupo'
    # , 'causabas_subcategoria'
    , 'FLAG_BASE'
    ]
]

df_sinasc.to_csv('base_suja/base_sinasc_suja.csv', index=False)

#----------------------------------------------- CNES ------------------------------------------------------------------
caminho = 'C:/Users/gabri/Documents/PSW_COVID_OBITO_FETAL/CNES'
variaveis = [
    'CNES'
    , 'TP_UNID'
    , 'def_tp_unid'
    , 'mun_MUNNOMEX'
    , 'uf_SIGLA_UF'
    , 'ano_competen'
    , 'mes_competen'
    , 'CENTROBS'
    , 'QTINST34'
    , 'QTINST35'
    , 'QTINST36'
    , 'QTINST37'
    , 'QTLEIT34'
    , 'QTLEIT38'
    , 'QTLEIT39'
    , 'QTLEIT40'
    , 'CENTRNEO'
]

df_cnes = func_apend_data(path=caminho, column=variaveis)
# Remover duplicadas
df_cnes = df_cnes.drop_duplicates()

# Separando o dicionário da variável TP_UNID
df_tp_unid = df_cnes[['TP_UNID', 'def_tp_unid']].drop_duplicates()
df_tp_unid.to_csv('dicionario_tp_unid_cnes.csv', index=False)

# Vericando missings nas variáveis de agregação
df_cnes['mun_MUNNOMEX'].isnull().sum()
df_cnes['uf_SIGLA_UF'].isnull().sum()
df_cnes['ano_competen'].isnull().sum()
df_cnes['mes_competen'].isnull().sum()
# Agregando a base
df_cnes_agreg = df_cnes.groupby(
 [
    'mun_MUNNOMEX'
    , 'uf_SIGLA_UF'
    , 'ano_competen'
    , 'mes_competen'
 ]
    , as_index=False
).agg(
    sum_CENTROBS=pd.NamedAgg(column='CENTROBS', aggfunc='sum')
    , sum_QTINST34=pd.NamedAgg(column='QTINST34', aggfunc='sum')
    , sum_QTINST35=pd.NamedAgg(column='QTINST35', aggfunc='sum')
    , sum_QTINST36=pd.NamedAgg(column='QTINST36', aggfunc='sum')
    , sum_QTINST37=pd.NamedAgg(column='QTINST37', aggfunc='sum')
    , sum_QTLEIT34=pd.NamedAgg(column='QTLEIT34', aggfunc='sum')
    , sum_QTLEIT38=pd.NamedAgg(column='QTLEIT38', aggfunc='sum')
    , sum_QTLEIT39=pd.NamedAgg(column='QTLEIT39', aggfunc='sum')
    , sum_QTLEIT40=pd.NamedAgg(column='QTLEIT40', aggfunc='sum')
    , sum_CENTRNEO=pd.NamedAgg(column='CENTRNEO', aggfunc='sum')
)

# Inserindo as dummies
df_cnes_pivot = df_cnes[
    [
        'mun_MUNNOMEX'
        , 'uf_SIGLA_UF'
        , 'ano_competen'
        , 'mes_competen'
        , 'TP_UNID'
    ]
]

# Pivot table usando dummies para contar ocorrências de TP_UNID
df_cnes_pivot = pd.pivot_table(df_cnes_pivot,
                             index=['mun_MUNNOMEX', 'uf_SIGLA_UF', 'ano_competen', 'mes_competen'],
                             columns='TP_UNID',
                             aggfunc='size',
                             fill_value=0).reset_index()

df_cnes_pivot.columns.name = None  # Remove o nome da coluna do índice
df_cnes_pivot.columns = ['TP_UNID_' + str(col) for col in df_cnes_pivot.columns]



df_cnes_agreg = df_cnes_agreg.merge(
    df_cnes_pivot
    , how='left'
    , left_on=['mun_MUNNOMEX', 'uf_SIGLA_UF', 'ano_competen', 'mes_competen']
    , right_on=['TP_UNID_mun_MUNNOMEX', 'TP_UNID_uf_SIGLA_UF', 'TP_UNID_ano_competen', 'TP_UNID_mes_competen']
)
df_cnes_agreg = df_cnes_agreg.drop(columns=['TP_UNID_mun_MUNNOMEX', 'TP_UNID_uf_SIGLA_UF', 'TP_UNID_ano_competen', 'TP_UNID_mes_competen'])
df_cnes_agreg.to_csv('base_suja/base_cnes_suja.csv', index=False)


