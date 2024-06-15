import pandas as pd

# joining SIM_DOFET E SISNASC
df_sim = pd.read_csv('base_suja/base_sim_dofet_suja.csv')
df_sinasc = pd.read_csv('base_suja/base_sinasc_suja.csv')

# Getting the names right
df_sim = df_sim.rename(
    columns={
        'data_obito': 'data_evento'
        , 'ano_obito': 'ano_evento'
        , 'ocor_MUNNOMEX': 'evento_MUNNOMEX'
        , 'ocor_CAPITAL': 'evento_CAPITAL'
        , 'ocor_REGIAO': 'evento_REGIAO'
        , 'ocor_SIGLA_UF': 'evento_SIGLA_UF'
    }
)

df_sinasc = df_sinasc.rename(
    columns={
        'data_nasc': 'data_evento'
        , 'ano_nasc': 'ano_evento'
        , 'nasc_MUNNOMEX': 'evento_MUNNOMEX'
        , 'nasc_CAPITAL': 'evento_CAPITAL'
        , 'nasc_REGIAO': 'evento_REGIAO'
        , 'nasc_SIGLA_UF': 'evento_SIGLA_UF'
    }
)

df_unificada = pd.concat([df_sim, df_sinasc])
df_unificada = df_unificada.reset_index(drop=True)
del df_sim, df_sinasc

# Merge CNES
df_unificada['mes_evento'] = [int(i.split('-')[1]) for i in df_unificada['data_evento']]
df_cnes = pd.read_csv('base_suja/base_cnes_suja.csv')

# antes do merge preciso tratar os missings nas colunas do join
# Vericando missings nas variáveis do merge
df_unificada['res_MUNNOMEX'].isnull().sum()
df_unificada['res_SIGLA_UF'].isnull().sum()
df_unificada['ano_evento'].isnull().sum()
df_unificada['mes_evento'].isnull().sum()

df_saida = df_unificada.merge(
    df_cnes
    , how='left'
    , left_on=['res_MUNNOMEX', 'ano_evento', 'mes_evento', 'res_SIGLA_UF']
    , right_on=['mun_MUNNOMEX', 'ano_competen', 'mes_competen', 'uf_SIGLA_UF']
)

df_saida.to_csv('base_suja/base_unificada_suja.csv', index=False)