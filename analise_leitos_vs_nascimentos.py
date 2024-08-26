import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from geopy.distance import geodesic

# df = pd.read_csv('base_suja/base_unificada_suja.csv')
#
# colunas = [
#     'data_evento'
#     , 'ano_evento'
#     , 'evento_MUNNOMEX'
#     , 'evento_SIGLA_UF'
#     , 'sum_CENTROBS'
#     , 'sum_QTINST34'
#     , 'sum_QTINST35'
#     , 'sum_QTINST36'
#     , 'sum_QTINST37'
#     , 'sum_QTLEIT34'
#     , 'sum_QTLEIT38'
#     , 'sum_QTLEIT39'
#     , 'sum_QTLEIT40'
#     , 'sum_CENTRNEO'
#     , 'TP_UNID_5'
#     , 'TP_UNID_7'
#     , 'TP_UNID_15'
#     , 'TP_UNID_36'
#     , 'FLAG_BASE'
# ]
# df = df[colunas]
#
# df['mes_ano_evento'] = [str(i)[0:7] for i in df['data_evento']]
#
# df_agrupado = df.groupby(
#  [
#     'evento_SIGLA_UF'
#     , 'evento_MUNNOMEX'
#     , 'ano_evento'
#  ]
#     , as_index=False
# ).agg(
#     QTINST34=pd.NamedAgg(column='sum_QTINST34', aggfunc='median')
#     , QTINST35=pd.NamedAgg(column='sum_QTINST35', aggfunc='median')
#     , QTINST36=pd.NamedAgg(column='sum_QTINST36', aggfunc='median')
#     , QTINST37=pd.NamedAgg(column='sum_QTINST37', aggfunc='median')
#     , QTLEIT34=pd.NamedAgg(column='sum_QTLEIT34', aggfunc='median')
#     , QTLEIT38=pd.NamedAgg(column='sum_QTLEIT38', aggfunc='median')
#     , QTLEIT39=pd.NamedAgg(column='sum_QTLEIT39', aggfunc='median')
#     , QTLEIT40=pd.NamedAgg(column='sum_QTLEIT40', aggfunc='median')
#     , TP_UNID_5=pd.NamedAgg(column='TP_UNID_5', aggfunc='median')
#     , TP_UNID_7=pd.NamedAgg(column='TP_UNID_7', aggfunc='median')
#     , TP_UNID_15=pd.NamedAgg(column='TP_UNID_15', aggfunc='median')
#     , TP_UNID_36=pd.NamedAgg(column='TP_UNID_36', aggfunc='median')
#     , QTD_NASCIMENTOS=pd.NamedAgg(column='data_evento', aggfunc='size')
#     )

# df_agrupado.to_csv('check_leitos_vs_nascimentos.csv', index=False)

# df_agrupado = pd.read_csv('./base_limpa/check_leitos_vs_nascimentos.csv')
#
# df_agrupado['QTINST'] = df_agrupado['QTINST34'] + df_agrupado['QTINST35'] + df_agrupado['QTINST36'] + df_agrupado['QTINST37']
# df_agrupado['QTLEIT'] = df_agrupado['QTLEIT34'] + df_agrupado['QTLEIT38'] + df_agrupado['QTLEIT39'] + df_agrupado['QTLEIT40']
#
# # Variação
# df_agrupado['var_QTD_NASCIMENTOS'] = df_agrupado.sort_values(
#   by=['evento_SIGLA_UF', 'evento_MUNNOMEX', 'ano_evento']
# ).groupby(
#   ['evento_SIGLA_UF', 'evento_MUNNOMEX']
# )['QTD_NASCIMENTOS'].pct_change()
#
# df_agrupado['var_QTINST'] = df_agrupado.sort_values(
#   by=['evento_SIGLA_UF', 'evento_MUNNOMEX', 'ano_evento']
# ).groupby(
#   ['evento_SIGLA_UF', 'evento_MUNNOMEX']
# )['QTINST'].pct_change()
#
# df_agrupado['var_QTLEIT'] = df_agrupado.sort_values(
#   by=['evento_SIGLA_UF', 'evento_MUNNOMEX', 'ano_evento']
# ).groupby(
#   ['evento_SIGLA_UF', 'evento_MUNNOMEX']
# )['QTLEIT'].pct_change()
#
# df_agrupado['var_TP_UNID_5'] = df_agrupado.sort_values(
#   by=['evento_SIGLA_UF', 'evento_MUNNOMEX', 'ano_evento']
# ).groupby(
#   ['evento_SIGLA_UF', 'evento_MUNNOMEX']
# )['TP_UNID_5'].pct_change()
#
# df_agrupado['var_TP_UNID_7'] = df_agrupado.sort_values(
#   by=['evento_SIGLA_UF', 'evento_MUNNOMEX', 'ano_evento']
# ).groupby(
#   ['evento_SIGLA_UF', 'evento_MUNNOMEX']
# )['TP_UNID_7'].pct_change()
#
# df_agrupado['var_TP_UNID_15'] = df_agrupado.sort_values(
#   by=['evento_SIGLA_UF', 'evento_MUNNOMEX', 'ano_evento']
# ).groupby(
#   ['evento_SIGLA_UF', 'evento_MUNNOMEX']
# )['TP_UNID_15'].pct_change()
#
# df_agrupado['var_TP_UNID_36'] = df_agrupado.sort_values(
#   by=['evento_SIGLA_UF', 'evento_MUNNOMEX', 'ano_evento']
# ).groupby(
#   ['evento_SIGLA_UF', 'evento_MUNNOMEX']
# )['TP_UNID_36'].pct_change()
#
# # Diferença
# df_agrupado['dif_QTD_NASCIMENTOS'] = df_agrupado.sort_values(
#   by=['evento_SIGLA_UF', 'evento_MUNNOMEX', 'ano_evento']
# ).groupby(
#   ['evento_SIGLA_UF', 'evento_MUNNOMEX']
# )['QTD_NASCIMENTOS'].diff()
#
# df_agrupado['dif_QTINST'] = df_agrupado.sort_values(
#   by=['evento_SIGLA_UF', 'evento_MUNNOMEX', 'ano_evento']
# ).groupby(
#   ['evento_SIGLA_UF', 'evento_MUNNOMEX']
# )['QTINST'].diff()
#
# df_agrupado['dif_QTLEIT'] = df_agrupado.sort_values(
#   by=['evento_SIGLA_UF', 'evento_MUNNOMEX', 'ano_evento']
# ).groupby(
#   ['evento_SIGLA_UF', 'evento_MUNNOMEX']
# )['QTLEIT'].diff()
#
# df_agrupado['dif_TP_UNID_5'] = df_agrupado.sort_values(
#   by=['evento_SIGLA_UF', 'evento_MUNNOMEX', 'ano_evento']
# ).groupby(
#   ['evento_SIGLA_UF', 'evento_MUNNOMEX']
# )['TP_UNID_5'].diff()
#
# df_agrupado['dif_TP_UNID_7'] = df_agrupado.sort_values(
#   by=['evento_SIGLA_UF', 'evento_MUNNOMEX', 'ano_evento']
# ).groupby(
#   ['evento_SIGLA_UF', 'evento_MUNNOMEX']
# )['TP_UNID_7'].diff()
#
# df_agrupado['dif_TP_UNID_15'] = df_agrupado.sort_values(
#   by=['evento_SIGLA_UF', 'evento_MUNNOMEX', 'ano_evento']
# ).groupby(
#   ['evento_SIGLA_UF', 'evento_MUNNOMEX']
# )['TP_UNID_15'].diff()
#
# df_agrupado['dif_TP_UNID_36'] = df_agrupado.sort_values(
#   by=['evento_SIGLA_UF', 'evento_MUNNOMEX', 'ano_evento']
# ).groupby(
#   ['evento_SIGLA_UF', 'evento_MUNNOMEX']
# )['TP_UNID_36'].diff()
#
# # Removendo os missings
# df_agrupado = df_agrupado.fillna(0)
#
# # Separando municípios que tiveram menos que 500 nascimentos em um ano
# df_agrupado['chave_mun_uf_evento'] = [f'{i}_{j}' for i, j in zip(df_agrupado['evento_MUNNOMEX'], df_agrupado['evento_SIGLA_UF'])]
# df_agrupado.to_csv('./base_limpa/check_leitos_vs_nascimentos.csv', index=False)

df_agrupado = pd.read_csv('./base_limpa/check_leitos_vs_nascimentos.csv')
df_nasc_menor_500 = df_agrupado[df_agrupado['QTD_NASCIMENTOS'] <= 500]
lista_mun_interesse = pd.unique(df_nasc_menor_500['chave_mun_uf_evento'])

# 4004 de um total de 4761
df_interesse = df_agrupado[df_agrupado['chave_mun_uf_evento'].isin(lista_mun_interesse)]
df_interesse = df_interesse.reset_index(drop=True)

colunas = [
    'evento_SIGLA_UF', 'evento_MUNNOMEX', 'ano_evento', 'QTINST', 'QTLEIT', 'TP_UNID_5', 'TP_UNID_7'
    , 'TP_UNID_15', 'TP_UNID_36', 'QTD_NASCIMENTOS', 'var_QTINST', 'var_QTLEIT', 'var_TP_UNID_5'
    , 'var_TP_UNID_7', 'var_TP_UNID_15', 'var_TP_UNID_36', 'var_QTD_NASCIMENTOS'
]
# Removendo os missings
df_interesse = df_interesse[colunas]

# Lista para armazenar as correlações todos os anos
correlacoes = []
ufs =  pd.unique(df_interesse['evento_SIGLA_UF'])
# Agrupar os dados por município e calcular a correlação
for uf in ufs:
    subset = df_interesse.loc[(df_interesse['evento_SIGLA_UF'] == uf) & (df_interesse['ano_evento'] != 2018)]
    if len(subset) > 1:  # Certifique-se de que há dados suficientes para calcular a correlação
        correlacao_QTLEIT = subset[['var_QTLEIT', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
        correlacao_QTINST = subset[['var_QTINST', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
        correlacao_TP_UNID_5 = subset[['var_TP_UNID_5', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
        correlacao_TP_UNID_7 = subset[['var_TP_UNID_7', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
        correlacao_TP_UNID_15 = subset[['var_TP_UNID_15', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
        correlacao_TP_UNID_36 = subset[['var_TP_UNID_36', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
        correlacoes.append({'evento_SIGLA_UF': uf, 'correlacao_QTLEIT': correlacao_QTLEIT
                           , 'correlacao_QTINST': correlacao_QTINST, 'correlacao_TP_UNID_5': correlacao_TP_UNID_5
                           , 'correlacao_TP_UNID_7': correlacao_TP_UNID_7, 'correlacao_TP_UNID_15': correlacao_TP_UNID_15
                           , 'correlacao_TP_UNID_36': correlacao_TP_UNID_36})

# Criar um DataFrame com os resultados
df_correlacoes = pd.DataFrame(correlacoes)
aa=df_correlacoes[(df_correlacoes['correlacao_QTLEIT'] >= 0.5) | (df_correlacoes['correlacao_QTINST'] >= 0.5) |
              (df_correlacoes['correlacao_TP_UNID_5'] >= 0.5) | (df_correlacoes['correlacao_TP_UNID_7'] >= 0.5) |
              (df_correlacoes['correlacao_TP_UNID_15'] >= 0.5) | (df_correlacoes['correlacao_TP_UNID_36'] >= 0.5)]

# Considerando variações positivas e negativas tenho somente uma uf que se observa euma forte correlação
aa

# grafico
# Criar gráficos de dispersão
#for uf in aa['evento_SIGLA_UF']:
#    subset = df_interesse[df_interesse['evento_SIGLA_UF'] == uf]
#    corr = round(df_correlacoes.loc[df_correlacoes['evento_SIGLA_UF'] == uf, 'correlacao_QTLEIT'], 2)
#    plt.figure(figsize=(10, 6))
#    plt.scatter(subset['var_QTLEIT'], subset['var_QTD_NASCIMENTOS'], alpha=0.5)
#    plt.title(f'Leitos VS Nasc corr {corr.reset_index(drop=True)[0]} municipio {uf} ')
#    plt.xlabel('Número de Leitos (var_QTDLEIT)')
#    plt.ylabel('Número de Nascimentos (var_qtd_nasc)')
#    plt.grid(True)
#    plt.show()

# Lista para armazenar as correlações todos os anos
correlacoes_ano = []
ufs =  pd.unique(df_interesse['evento_SIGLA_UF'])
# Agrupar os dados por município e calcular a correlação
for uf in ufs:
    for ano in [2019, 2020, 2021, 2022]:
        subset = df_interesse.loc[(df_interesse['evento_SIGLA_UF'] == uf) & (df_interesse['ano_evento'] == ano)]
        if len(subset) > 1:  # Certifique-se de que há dados suficientes para calcular a correlação
            correlacao_QTLEIT = subset[['var_QTLEIT', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
            correlacao_QTINST = subset[['var_QTINST', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
            correlacao_TP_UNID_5 = subset[['var_TP_UNID_5', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
            correlacao_TP_UNID_7 = subset[['var_TP_UNID_7', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
            correlacao_TP_UNID_15 = subset[['var_TP_UNID_15', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
            correlacao_TP_UNID_36 = subset[['var_TP_UNID_36', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
            correlacoes_ano.append({'evento_SIGLA_UF': uf, 'ano_evento': ano,'correlacao_QTLEIT': correlacao_QTLEIT
                               , 'correlacao_QTINST': correlacao_QTINST, 'correlacao_TP_UNID_5': correlacao_TP_UNID_5
                               , 'correlacao_TP_UNID_7': correlacao_TP_UNID_7, 'correlacao_TP_UNID_15': correlacao_TP_UNID_15
                               , 'correlacao_TP_UNID_36': correlacao_TP_UNID_36})

# Criar um DataFrame com os resultados
df_correlacoes_ano = pd.DataFrame(correlacoes_ano)
bb=df_correlacoes_ano[(df_correlacoes_ano['correlacao_QTLEIT'] >= 0.5) | (df_correlacoes_ano['correlacao_QTINST'] >= 0.5) |
              (df_correlacoes_ano['correlacao_TP_UNID_5'] >= 0.5) | (df_correlacoes_ano['correlacao_TP_UNID_7'] >= 0.5) |
              (df_correlacoes_ano['correlacao_TP_UNID_15'] >= 0.5) | (df_correlacoes_ano['correlacao_TP_UNID_36'] >= 0.5)]

# agora vou considerar somente as variações negativas
# Lista para armazenar as correlações todos os anos
correlacoes = []
ufs =  pd.unique(df_interesse['evento_SIGLA_UF'])
# Agrupar os dados por município e calcular a correlação
for uf in ufs:
    subset = df_interesse.loc[(df_interesse['evento_SIGLA_UF'] == uf) & (df_interesse['ano_evento'] != 2018) &
                              (df_interesse['var_QTD_NASCIMENTOS'] < 0)]
    if len(subset) > 1:  # Certifique-se de que há dados suficientes para calcular a correlação
        correlacao_QTLEIT = subset[['var_QTLEIT', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
        correlacao_QTINST = subset[['var_QTINST', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
        correlacao_TP_UNID_5 = subset[['var_TP_UNID_5', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
        correlacao_TP_UNID_7 = subset[['var_TP_UNID_7', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
        correlacao_TP_UNID_15 = subset[['var_TP_UNID_15', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
        correlacao_TP_UNID_36 = subset[['var_TP_UNID_36', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
        correlacoes.append({'evento_SIGLA_UF': uf, 'correlacao_QTLEIT': correlacao_QTLEIT
                           , 'correlacao_QTINST': correlacao_QTINST, 'correlacao_TP_UNID_5': correlacao_TP_UNID_5
                           , 'correlacao_TP_UNID_7': correlacao_TP_UNID_7, 'correlacao_TP_UNID_15': correlacao_TP_UNID_15
                           , 'correlacao_TP_UNID_36': correlacao_TP_UNID_36})

# Criar um DataFrame com os resultados
df_correlacoes = pd.DataFrame(correlacoes)
aa_=df_correlacoes[(df_correlacoes['correlacao_QTLEIT'] >= 0.5) | (df_correlacoes['correlacao_QTINST'] >= 0.5) |
              (df_correlacoes['correlacao_TP_UNID_5'] >= 0.5) | (df_correlacoes['correlacao_TP_UNID_7'] >= 0.5) |
              (df_correlacoes['correlacao_TP_UNID_15'] >= 0.5) | (df_correlacoes['correlacao_TP_UNID_36'] >= 0.5)]

# Lista para armazenar as correlações todos os anos e somente variacoes negativas
correlacoes_ano = []
ufs =  pd.unique(df_interesse['evento_SIGLA_UF'])
# Agrupar os dados por município e calcular a correlação
for uf in ufs:
    for ano in [2019, 2020, 2021, 2022]:
        subset = df_interesse.loc[(df_interesse['evento_SIGLA_UF'] == uf) & (df_interesse['ano_evento'] == ano) &
                                 (df_interesse['var_QTD_NASCIMENTOS'] < 0)]
        if len(subset) > 1:  # Certifique-se de que há dados suficientes para calcular a correlação
            correlacao_QTLEIT = subset[['var_QTLEIT', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
            correlacao_QTINST = subset[['var_QTINST', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
            correlacao_TP_UNID_5 = subset[['var_TP_UNID_5', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
            correlacao_TP_UNID_7 = subset[['var_TP_UNID_7', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
            correlacao_TP_UNID_15 = subset[['var_TP_UNID_15', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
            correlacao_TP_UNID_36 = subset[['var_TP_UNID_36', 'var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
            correlacoes_ano.append({'evento_SIGLA_UF': uf, 'ano_evento': ano,'correlacao_QTLEIT': correlacao_QTLEIT
                               , 'correlacao_QTINST': correlacao_QTINST, 'correlacao_TP_UNID_5': correlacao_TP_UNID_5
                               , 'correlacao_TP_UNID_7': correlacao_TP_UNID_7, 'correlacao_TP_UNID_15': correlacao_TP_UNID_15
                               , 'correlacao_TP_UNID_36': correlacao_TP_UNID_36})

# Criar um DataFrame com os resultados
df_correlacoes_ano = pd.DataFrame(correlacoes_ano)
bb_=df_correlacoes_ano[(df_correlacoes_ano['correlacao_QTLEIT'] >= 0.5) | (df_correlacoes_ano['correlacao_QTINST'] >= 0.5) |
              (df_correlacoes_ano['correlacao_TP_UNID_5'] >= 0.5) | (df_correlacoes_ano['correlacao_TP_UNID_7'] >= 0.5) |
              (df_correlacoes_ano['correlacao_TP_UNID_15'] >= 0.5) | (df_correlacoes_ano['correlacao_TP_UNID_36'] >= 0.5)]
########################################################################################################################
# Verificar pra onde foi os nascimentos?
########################################################################################################################
#Variação entre municípios

# Calcular a distância entre municipios selecionar os mais próximos dos municípios que tiveram queda e
# verificar se tiveram aumento no número de nascimentos.
base_lat_long_mun = pd.read_csv('base_limpa/latitude_longitude_municipios.csv', sep=';', decimal=',')

base_lat_long_mun['mun_uf'] = [f'{i}_{j}' for i, j in zip(base_lat_long_mun['mun_MUNNOMEX'], base_lat_long_mun['uf_SIGLA_UF'])]

# Obtendo as UFs únicas
ufs = pd.unique(base_lat_long_mun['uf_SIGLA_UF'])
lista = []

# Iterando por cada UF
for uf in ufs:
    df_uf = base_lat_long_mun[base_lat_long_mun['uf_SIGLA_UF'] == uf]
    munic = pd.unique(df_uf['mun_uf']).tolist()
    pares_processados = set()  # Usado para armazenar pares já processados
    # Iterando por cada município
    for i, mun in enumerate(munic):
        for mun_2 in munic[i+1:]:
            print(f'{uf} Municipio 1:{mun} Municipio 2 {mun_2}')
            # Verificando se o par já foi processado
            par = tuple(sorted([mun, mun_2]))
            if par in pares_processados:
                continue

            # Coordenadas do primeiro município
            lat1 =df_uf.loc[df_uf['mun_uf'] == mun, 'mun_LATITUDE'].values[0]
            long1 = df_uf.loc[df_uf['mun_uf'] == mun, 'mun_LONGITUDE'].values[0]

            # Coordenadas do segundo município
            lat2 = df_uf.loc[df_uf['mun_uf'] == mun_2, 'mun_LATITUDE'].values[0]
            long2 = df_uf.loc[df_uf['mun_uf'] == mun_2, 'mun_LONGITUDE'].values[0]

            # Calculando a distância
            distancia = geodesic((lat1, long1), (lat2, long2)).kilometers

            # Criando o dicionário com os dados do par
            dicionario = {
                'municipio1':[mun], 'municipio2': [mun_2], 'dist_km': [distancia],
                'lat_mun1': [lat1], 'long_mun1': [long1],
                'lat_mun2': [lat2], 'long_mun2': [long2]
            }
            lista.append(dicionario)

            # Adicionando o par ao conjunto de pares processados
            pares_processados.add(par)

lista = [pd.DataFrame(i) for i in lista]
lista = pd.concat(lista)

# lista.to_csv('distancia_municipios_uf_intra.csv', sep=';', decimal=',')

# pares de  municipios vizinhos
pares = [
['AC', 'AM']
,['AC', 'RO']
,['AL', 'BA']
,['AL', 'PE']
,['AL', 'SE']
,['AM', 'MT']
,['AM', 'PA']
,['AM', 'RO']
,['AM', 'RR']
,['AP', 'PA']
,['BA', 'ES']
,['BA', 'GO']
,['BA', 'MG']
,['BA', 'PE']
,['BA', 'PI']
,['BA', 'TO']
,['CE', 'PB']
,['CE', 'PE']
,['CE', 'PI']
,['CE', 'RN']
,['DF', 'GO']
,['DF', 'MG']
,['ES', 'MG']
,['ES', 'RJ']
,['GO', 'MG']
,['GO', 'MT']
,['GO', 'MS']
,['GO', 'TO']
,['MA', 'PA']
,['MA', 'PI']
,['MA', 'TO']
,['MG', 'MT']
,['MG', 'RJ']
,['MG', 'SP']
,['MS', 'MT']
,['MS', 'PR']
,['MS', 'SP']
,['MT', 'PA']
,['MT', 'RO']
,['MT', 'TO']
,['PA', 'TO']
,['PB', 'PE']
,['PB', 'RN']
,['PE', 'PI']
,['PI', 'TO']
,['PR', 'SC']
,['PR', 'SP']
,['RJ', 'SP']
,['RS', 'SC']]

# Obtendo as UFs únicas
lista_pares = []

# Iterando por cada UF
for par in pares:
    df_uf1 = base_lat_long_mun[base_lat_long_mun['uf_SIGLA_UF'] == par[0]]
    df_uf2 = base_lat_long_mun[base_lat_long_mun['uf_SIGLA_UF'] == par[1]]
    munic1 = pd.unique(df_uf1['mun_uf']).tolist()
    munic2 = pd.unique(df_uf2['mun_uf']).tolist()
    pares_processados = set()  # Usado para armazenar pares já processados
    # Iterando por cada município
    for mun1 in munic1:
        for mun2 in munic2:
            print(f'Par {par} Municipio 1:{mun1} Municipio 2 {mun2}')
            # Coordenadas do primeiro município
            lat1 =df_uf1.loc[df_uf1['mun_uf'] == mun1, 'mun_LATITUDE'].values[0]
            long1 = df_uf1.loc[df_uf1['mun_uf'] == mun1, 'mun_LONGITUDE'].values[0]

            # Coordenadas do segundo município
            lat2 = df_uf2.loc[df_uf2['mun_uf'] == mun2, 'mun_LATITUDE'].values[0]
            long2 = df_uf2.loc[df_uf2['mun_uf'] == mun2, 'mun_LONGITUDE'].values[0]

            # Calculando a distância
            distancia = geodesic((lat1, long1), (lat2, long2)).kilometers

            # Criando o dicionário com os dados do par
            dicionario = {
                'municipio1':[mun1], 'municipio2': [mun2], 'dist_km': [distancia],
                'lat_mun1': [lat1], 'long_mun1': [long1],
                'lat_mun2': [lat2], 'long_mun2': [long2]
            }
            lista_pares.append(dicionario)
    lista_pares = [pd.DataFrame(i) for i in lista_pares]
    lista_pares = pd.concat(lista_pares)
    lista_pares.to_csv(f'./ufs_vizinhas/distancia_municipios_uf_vizinhos_{par[0]}_{par[1]}.csv', sep=';', decimal=',')

# Junta os arquivos das distâncias
path='./ufs_vizinhas/'
arq = os.listdir(path)
df_distancia_mun = [pd.read_csv(f'{path}{i}', sep=';', decimal=',') for i in arq]
df_distancia_mun = pd.concat(df_distancia_mun)
# df_distancia_mun.to_csv('./base_limpa/distancia_municipios.csv', sep=';', decimal=',',  index=False)

lista = pd.read_csv('./base_limpa/distancia_municipios.csv', sep=';', decimal=',')
lista = lista.drop(columns='Unnamed: 0').reset_index(drop=True)
lista = lista.drop_duplicates().reset_index(drop=True)
munic1 = lista['municipio1']
munic2 = lista['municipio2']
lista2 = lista.copy()
lista2['municipio1']=munic2
lista2['municipio2']=munic1
lista3 = pd.concat([lista, lista2]).reset_index(drop=True)

del lista, lista2

# selecionando somente municípios com até 100 km de distância
lista_100 = lista3[lista3['dist_km'] <= 100.99].sort_values(['municipio1', 'dist_km']).reset_index(drop=True)
#del lista3
variacoes_negativas = df_agrupado.loc[(df_agrupado['ano_evento'] != 2018) &
                              (df_agrupado['var_QTD_NASCIMENTOS'] < 0)].reset_index(drop=True)

# Vou pegar a chave
chave_uf_mun = pd.unique(variacoes_negativas['chave_mun_uf_evento'])
colunas = [
    'chave_mun_uf_evento'
    , 'ano_evento'
    , 'QTD_NASCIMENTOS'
    , 'QTINST'
    , 'QTLEIT'
    , 'TP_UNID_5'
    , 'TP_UNID_7'
    , 'TP_UNID_15'
    , 'TP_UNID_36'
]
contador = 1
lista_perde_ganha = []
for munic in chave_uf_mun:
    print(f'Rodada {contador} de {len(chave_uf_mun)}')
    contador+=1
    muncipio_transf = lista_100.loc[lista_100['municipio1']==munic, 'municipio2']
    df_perde = df_agrupado.loc[df_agrupado['chave_mun_uf_evento'] == munic, colunas].reset_index(drop=True)
    df_ganha = df_agrupado.loc[df_agrupado['chave_mun_uf_evento'].isin(muncipio_transf), colunas].reset_index(drop=True)
    # Definir todos os anos e municípios esperados
    anos = [2018, 2019, 2020, 2021, 2022]
    municipios_perde = df_perde['chave_mun_uf_evento'].unique()
    municipios_ganha = df_ganha['chave_mun_uf_evento'].unique()
    # Criar um MultiIndex com todas as combinações possíveis de anos e municípios
    idx_perde = pd.MultiIndex.from_product([anos, municipios_perde], names=['ano_evento', 'chave_mun_uf_evento'])
    # Reindexar o DataFrame para incluir todas as combinações e preencher valores faltantes com 0
    df_perde = df_perde.set_index(['ano_evento', 'chave_mun_uf_evento']).reindex(idx_perde, fill_value=0).reset_index()

    # Criar um MultiIndex com todas as combinações possíveis de anos e municípios
    idx_ganha = pd.MultiIndex.from_product([anos, municipios_ganha], names=['ano_evento', 'chave_mun_uf_evento'])
    # Reindexar o DataFrame para incluir todas as combinações e preencher valores faltantes com 0
    df_ganha = df_ganha.set_index(['ano_evento', 'chave_mun_uf_evento']).reindex(idx_ganha, fill_value=0).reset_index()


    df_ganha.columns = [f'compara_{i}' for i in df_ganha.columns]
    df_perde = df_perde.merge(df_ganha, how='left', left_on='ano_evento', right_on='compara_ano_evento')
    lista_perde_ganha.append(df_perde)

df_perde_ganha = pd.concat(lista_perde_ganha)
del lista_perde_ganha
df_perde_ganha = df_perde_ganha[~df_perde_ganha['compara_chave_mun_uf_evento'].isnull()].reset_index(drop=True)
contador = 0
lista_perde_ganha_cor = []
for munic in chave_uf_mun:
    contador += 1
    df = df_perde_ganha[df_perde_ganha['chave_mun_uf_evento'] == munic]
    munic_ganha = pd.unique(df['compara_chave_mun_uf_evento'])
    contador2 = 1
    for munic2 in munic_ganha:
        print(f'Rodada {contador} de {len(chave_uf_mun)} munic perde {munic} munic ganha {munic2} subrodada {contador2} de {len(munic_ganha)}')
        contador2 += 1
        df_cor = df[df['compara_chave_mun_uf_evento']==munic2].sort_values('ano_evento')
        if len(df_cor) == 5:
            # municipio perde
            df_cor['var_QTD_NASCIMENTOS'] = df_cor.sort_values(by=['ano_evento']).groupby(['chave_mun_uf_evento'])['QTD_NASCIMENTOS'].pct_change()
            df_cor['var_QTINST'] = df_cor.sort_values(by=['ano_evento']).groupby(['chave_mun_uf_evento'])['QTINST'].pct_change()
            df_cor['var_QTLEIT'] = df_cor.sort_values(by=['ano_evento']).groupby(['chave_mun_uf_evento'])['QTLEIT'].pct_change()
            df_cor['var_TP_UNID_5'] = df_cor.sort_values(by=['ano_evento']).groupby(['chave_mun_uf_evento'])['TP_UNID_5'].pct_change()
            df_cor['var_TP_UNID_7'] = df_cor.sort_values(by=['ano_evento']).groupby(['chave_mun_uf_evento'])['TP_UNID_7'].pct_change()
            df_cor['var_TP_UNID_15'] = df_cor.sort_values(by=['ano_evento']).groupby(['chave_mun_uf_evento'])['TP_UNID_15'].pct_change()
            df_cor['var_TP_UNID_36'] = df_cor.sort_values(by=['ano_evento']).groupby(['chave_mun_uf_evento'])['TP_UNID_36'].pct_change()

            df_cor['dif_QTD_NASCIMENTOS'] = df_cor.sort_values(by=['ano_evento']).groupby(['chave_mun_uf_evento'])['QTD_NASCIMENTOS'].diff()
            df_cor['dif_QTINST'] = df_cor.sort_values(by=['ano_evento']).groupby(['chave_mun_uf_evento'])['QTINST'].diff()
            df_cor['dif_QTLEIT'] = df_cor.sort_values(by=['ano_evento']).groupby(['chave_mun_uf_evento'])['QTLEIT'].diff()
            df_cor['dif_TP_UNID_5'] = df_cor.sort_values(by=['ano_evento']).groupby(['chave_mun_uf_evento'])['TP_UNID_5'].diff()
            df_cor['dif_TP_UNID_7'] = df_cor.sort_values(by=['ano_evento']).groupby(['chave_mun_uf_evento'])['TP_UNID_7'].diff()
            df_cor['dif_TP_UNID_15'] = df_cor.sort_values(by=['ano_evento']).groupby(['chave_mun_uf_evento'])['TP_UNID_15'].diff()
            df_cor['dif_TP_UNID_36'] = df_cor.sort_values(by=['ano_evento']).groupby(['chave_mun_uf_evento'])['TP_UNID_36'].diff()

            # municipio ganha
            df_cor['compara_var_QTD_NASCIMENTOS'] = df_cor.sort_values(by=['compara_ano_evento']).groupby(['compara_chave_mun_uf_evento'])['compara_QTD_NASCIMENTOS'].pct_change()
            df_cor['compara_var_QTINST'] =  df_cor.sort_values(by=['compara_ano_evento']).groupby(['compara_chave_mun_uf_evento'])['compara_QTINST'].pct_change()
            df_cor['compara_var_QTLEIT'] = df_cor.sort_values(by=['compara_ano_evento']).groupby(['compara_chave_mun_uf_evento'])['compara_QTLEIT'].pct_change()
            df_cor['compara_var_TP_UNID_5'] = df_cor.sort_values(by=['compara_ano_evento']).groupby(['compara_chave_mun_uf_evento'])['compara_TP_UNID_5'].pct_change()
            df_cor['compara_var_TP_UNID_7'] = df_cor.sort_values(by=['compara_ano_evento']).groupby(['compara_chave_mun_uf_evento'])['compara_TP_UNID_7'].pct_change()
            df_cor['compara_var_TP_UNID_15'] = df_cor.sort_values(by=['compara_ano_evento']).groupby(['compara_chave_mun_uf_evento'])['compara_TP_UNID_15'].pct_change()
            df_cor['compara_var_TP_UNID_36'] = df_cor.sort_values(by=['compara_ano_evento']).groupby(['compara_chave_mun_uf_evento'])['compara_TP_UNID_36'].pct_change()

            df_cor['compara_dif_QTD_NASCIMENTOS'] = df_cor.sort_values(by=['compara_ano_evento']).groupby(['compara_chave_mun_uf_evento'])['compara_QTD_NASCIMENTOS'].diff()
            df_cor['compara_dif_QTINST'] = df_cor.sort_values(by=['compara_ano_evento']).groupby(['compara_chave_mun_uf_evento'])['compara_QTINST'].diff()
            df_cor['compara_dif_QTLEIT'] = df_cor.sort_values(by=['compara_ano_evento']).groupby(['compara_chave_mun_uf_evento'])['compara_QTLEIT'].diff()
            df_cor['compara_dif_TP_UNID_5'] = df_cor.sort_values(by=['compara_ano_evento']).groupby(['compara_chave_mun_uf_evento'])['compara_TP_UNID_5'].diff()
            df_cor['compara_dif_TP_UNID_7'] = df_cor.sort_values(by=['compara_ano_evento']).groupby(['compara_chave_mun_uf_evento'])['compara_TP_UNID_7'].diff()
            df_cor['compara_dif_TP_UNID_15'] = df_cor.sort_values(by=['compara_ano_evento']).groupby(['compara_chave_mun_uf_evento'])['compara_TP_UNID_15'].diff()
            df_cor['compara_dif_TP_UNID_36'] = df_cor.sort_values(by=['compara_ano_evento']).groupby(['compara_chave_mun_uf_evento'])['compara_TP_UNID_36'].diff()
            # Correlações
            df_cor = df_cor[df_cor['ano_evento']!=2018]
            df_cor['cor_var_nasc'] = df_cor[['var_QTD_NASCIMENTOS', 'compara_var_QTD_NASCIMENTOS']].corr().iloc[0, 1]
            df_cor['cor_dif_nasc'] = df_cor[['dif_QTD_NASCIMENTOS', 'compara_dif_QTD_NASCIMENTOS']].corr().iloc[0, 1]
            lista_perde_ganha_cor.append(df_cor)
        else:
            sys.exit(f'Rodada {contador} de {len(chave_uf_mun)} munic perde {munic} munic ganha {munic2} subrodada {contador2} de {len(munic_ganha)}')

df_perde_ganha_cor = pd.concat(lista_perde_ganha_cor).reset_index(drop=True)

df_perde_ganha_cor_ = df_perde_ganha_cor.merge(lista3, how='left', left_on=['chave_mun_uf_evento', 'compara_chave_mun_uf_evento']
                                               , right_on=['municipio1',  'municipio2'])

df_perde_ganha_cor_.to_csv('./df_mun_proximos_cor_nascimentos_com_distancia.csv', index=False)




