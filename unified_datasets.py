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
    }
)

df_sinasc = df_sinasc.rename(
    columns={
        'data_nasc': 'data_evento'
        , 'ano_nasc': 'ano_evento'
        , 'nasc_MUNNOMEX': 'evento_MUNNOMEX'
        , 'nasc_CAPITAL': 'evento_CAPITAL'
        , 'nasc_REGIAO': 'evento_REGIAO'
    }
)

df_unificada = pd.concat([df_sim, df_sinasc])
df_unificada.reset_index(drop=True)
del df_sim, df_sinasc

# Merge CNES
df_unificada['mes_evento']=[int(i) for i in df_unificada['data_evento']]