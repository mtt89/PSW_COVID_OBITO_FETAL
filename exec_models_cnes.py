#Importing Modules
from Funcoes_auxiliares.func_aux import *
import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from scipy import stats
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#from google.colab import drive
from pandas.testing import assert_frame_equal
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from tabulate import tabulate
import warnings
from Funcoes_auxiliares.func_aux import *
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.metrics import classification_report, confusion_matrix
import time
warnings.filterwarnings("ignore")

df = pd.read_csv('base_limpa/base_unificada_limpa_remocao.csv')
# df = pd.read_csv('base_limpa/base_unificada_limpa_com_input.csv')
########################################################################################################################
# Trabalhando a informação de leitos
# Somando leitos
df['QTINST'] = df['sum_QTINST34'] + df['sum_QTINST35'] + df['sum_QTINST36'] + df['sum_QTINST37'] #0 ate 15, ate 50, > 50
df['QTLEIT'] = df['sum_QTLEIT34'] + df['sum_QTLEIT38'] + df['sum_QTLEIT39'] + df['sum_QTLEIT40'] # 0 ate 15, ate 50, > 50
df['QT_TP_UNID'] = df['TP_UNID_5'] + df['TP_UNID_7'] + df['TP_UNID_15'] + df['TP_UNID_36'] # 0 ate 15, ate 50, > 50

def func_leitos(qtd):
    if qtd == 0:
        return 'zero'
    elif 0 < qtd <= 15:
        return 'entre_1_15'
    elif 15 < qtd <= 50:
        return 'entre_15_50'
    elif qtd > 50:
        return 'maior_50'
    else:
        return 'ignorado'

df['cat_QTINST'] = [func_leitos(qtd=i) for i in df['QTINST']]
df['cat_QTLEIT'] = [func_leitos(qtd=i) for i in df['QTLEIT']]
df['cat_QT_TP_UNID'] = [func_leitos(qtd=i) for i in df['QT_TP_UNID']]

# # Pop fem censo 2022
# base_pop = pd.read_csv('./pop_fem_2022/pop_fem_censo_2022_ibge.csv', encoding='ISO-8859-1', sep = ';')
# base_pop['municipio_sep'] = [i.split('(')[0] for i in base_pop['municipio']]
# base_pop['UF'] = [i.split('(')[1].replace(')', '') for i in base_pop['municipio']]
#
# base_pop['municipio_limpo'] = [func_limpar_string(i).replace("'", "").replace("-", " ") for i in base_pop['municipio_sep']]
#
# junta = base_pop[['municipio_limpo', 'UF', 'Pop_feminina_entre_10_50_anos_censo_2022']]
#
# df['evento_MUNNOMEX_limpo'] = [i.replace("'", '').replace("-", " ") for i in df['evento_MUNNOMEX']]
#
# df['evento_MUNNOMEX_limpo'] = np.where(df['evento_MUNNOMEX_limpo']=='EMBU', 'EMBU DAS ARTES', df['evento_MUNNOMEX_limpo'])
# df['evento_MUNNOMEX_limpo'] = np.where(df['evento_MUNNOMEX_limpo']=='PARATI', 'PARATY', df['evento_MUNNOMEX_limpo'])
# df['evento_MUNNOMEX_limpo'] = np.where(df['evento_MUNNOMEX_limpo']== 'MOJI MIRIM',  'MOGI MIRIM', df['evento_MUNNOMEX_limpo'])
# df['evento_MUNNOMEX_limpo'] = np.where(df['evento_MUNNOMEX_limpo']== 'ITAPAGE',  'ITAPAJE', df['evento_MUNNOMEX_limpo'])
# df['evento_MUNNOMEX_limpo'] = np.where(df['evento_MUNNOMEX_limpo']== 'ITAPAGE',  'ITAPAJE', df['evento_MUNNOMEX_limpo'])
#
# df = df.merge(junta, how='left', left_on=['evento_MUNNOMEX_limpo', 'evento_SIGLA_UF'], right_on=['municipio_limpo', 'UF'])
#
# df.loc[df['evento_MUNNOMEX_limpo'].str.contains('SANTA ISABEL DO PARA', case=False, na=False), ['evento_MUNNOMEX_limpo',  'evento_SIGLA_UF']]
#
# aa=df[df['Pop_feminina_entre_10_50_anos_censo_2022'].isnull()]
#
# pd.unique(aa['evento_MUNNOMEX'])
########################################################################################################################

# Peso calculado
df['cat_peso_calc'] = [
    func_peso_calculado(sexo, peso, int(round(semana_gest,0))) for sexo, peso, semana_gest in zip(df['SEXO'], df['PESO'], df['SEMAGESTAC'])
]

df['cat_peso_calc'].value_counts()

aa = df.value_counts(['SEXO', 'def_sexo'])

# Função para filtrar o dataframe e adicionar a coluna 'ANO'
def filtrar_e_adicionar(df, per):
    anos = list(map(int, per.split('_')))
    df_ano = df.loc[df['ano_evento'].isin(anos)]
    df_ano = df_ano.reset_index(drop=True)
    df_ano['ANO'] = [0 if i == anos[0] else 1 for i in df_ano['ano_evento']]
    return df_ano

lista_periodo = [
    '2019_2020', '2019_2021', '2019_2022', '2020_2021', '2020_2022', '2021_2022', '2019_2020_2021'
    , '2019_2020_2022', '2019_2021_2022', '2020_2021_2022', '2019_2020_2021_2022'
    ]

# Exemplos de uso
for periodo in lista_periodo:
    missing = 'missing_removido' #'missing_com_input'
    df_ano = filtrar_e_adicionar(df, periodo)
    df_ano['ano_evento'].value_counts()
    df_ano['cat_peso_calc'].value_counts()

    # Separando df_mod
    variaveis_1 = [
         'ANO'
        , 'evento_REGIAO'
        , 'idademae_faixa'
        , 'escolaridade_mae'
        , 'tipo_gravidez'
        , 'idade_gestacao_faixa'
        , 'def_sexo'
        , 'peso_faixa'
        , 'cat_peso_calc'
        , 'FLAG_BASE'
        ,
        ]
    df_mod = df_ano[variaveis_1]

    # Dummies
    df_mod = pd.get_dummies(df_mod)

    # OBITO contagem
    NPT = (df_mod['FLAG_BASE_SIM_DOFET']==1).sum()
    NT = (df_mod['FLAG_BASE_SIM_DOFET']==0).sum()
    #Defining covariates
    var_model=[
         'evento_REGIAO_Centro-Oeste'
        , 'evento_REGIAO_Nordeste'
        , 'evento_REGIAO_Norte'
        #, 'evento_REGIAO_Sudeste'
        , 'evento_REGIAO_Sul'
        #, 'idademae_faixa_entre_20_34'
        , 'idademae_faixa_entre_35_39'
        , 'idademae_faixa_maior_igual_40'
        , 'idademae_faixa_menor_igual_19'
        , 'escolaridade_mae_Ensino_medio'
        #, 'escolaridade_mae_Ensino_superior'
        , 'escolaridade_mae_Fundamental'
        , 'escolaridade_mae_Sem_escolaridade'
        , 'tipo_gravidez_Multipla'
        #, 'tipo_gravidez_Unica'
        #, 'PRM'
        #, 'def_sexo_Feminino'
        , 'def_sexo_Masculino'
        #, 'cat_peso_calc_AIG'
        , 'cat_peso_calc_GIG'
        , 'cat_peso_calc_PIG'
        #, 'peso_faixa_entre_1500_2499'
        #, 'peso_faixa_entre_2500_3500'
        #, 'peso_faixa_entre_3500_3999'
        #, 'peso_faixa_entre_500_1499'
        #, 'peso_faixa_maior_igual_4000'
        #, 'peso_faixa_menor_500'
        #,'cat_CENTROBS'
        , 'cat_QTINST34'
        ,'cat_QTINST35'
        ,'cat_QTINST36'
        , 'cat_QTINST37'
        , 'cat_QTLEIT34'
        , 'cat_QTLEIT38'
        , 'cat_QTLEIT39'
        , 'cat_QTLEIT40'
        , 'cat_CENTRNEO'
        , 'cat_TP_UNID_5'
        , 'cat_TP_UNID_7'
        , 'cat_TP_UNID_15'
        , 'cat_TP_UNID_36'
    ]

########################################################################################################################
    print("""
    Modelo 0
    Missing: Removido
    balanceamento: Sem balancear
    Tipo de Balancemaneto:
    """)
    # Primeiro Modelo

    X = df_mod[var_model].values

    ## FIRST REGRESSION
    y = df_mod[['ANO']].values
    clf = LogisticRegression(random_state=0, max_iter=2000).fit(X, y)
    df_mod = df_mod.assign(PROPENSITY_SCORE=clf.predict_proba(X)[:, 1])

    #USING PROPENSITY SCORE TO SELECT SAMPLES TO SECOND REGRESSION
    psw_base=df_mod[((df_mod['PROPENSITY_SCORE']>df_mod['PROPENSITY_SCORE'].quantile(0.1)) &
                     (df_mod['PROPENSITY_SCORE']<df_mod['PROPENSITY_SCORE'].quantile(0.9)))]

    #COUNTING
    po=len(df_mod)
    tpo=len(df_mod[df_mod['ANO']==1])
    cpo=len(df_mod[df_mod['ANO']==0])
    pa=len(psw_base)
    tpa=len(psw_base[psw_base['ANO']==1])
    cpa=len(psw_base[psw_base['ANO']==0])

    print('----------------------------------------------------------------')
    print('USING PROPENSITY SCORE TO SELECT/MATCH SAMPLES')
    print('----------------------------------------------------------------')
    print('N without missing  :', po)
    print('Treated samples    :', tpo,np.round(100*tpo/po,2),'%')
    print('Controled samples  :', cpo,np.round(100*cpo/po,2),'%')
    print('----------------------------------------------------------------')
    print('SELECTED/MATCHET SAMPLES')
    print('----------------------------------------------------------------')
    print('% Selected         :', np.round(100*pa/po,2),'%')
    print('N selected         :', pa)
    print('Treated selected   :', tpa,np.round(100*tpa/pa,2),'%')
    print('Controled selected :', cpa,np.round(100*cpa/pa,2),'%')
    print('----------------------------------------------------------------')
    print('')

    ## SECOND REGRESSION

    aux = ['ANO']
    w = psw_base['ANO']/psw_base['PROPENSITY_SCORE'] + ((1-psw_base['ANO'])/(1-psw_base['PROPENSITY_SCORE']))

    X = psw_base[aux+var_model].astype(float).values
    y = psw_base[['FLAG_BASE_SIM_DOFET']].values

    X_ANO = sm.add_constant(X)
    clf_ano = sm.Logit(y, X_ANO, weights=w).fit(maxiter=1000)
    aux=['Intercept','ANO']

    print('----------------------------------------------------------------')
    print('PSW REPORT',periodo)
    print('----------------------------------------------------------------')
    print(clf_ano.summary(xname=aux+var_model))
    print('----------------------------------------------------------------')

    IC=np.exp(clf_ano.conf_int(0.05))
    odds_ratio=pd.DataFrame(
        data = {
            'Var':aux+var_model
            , 'Odds_ratio': np.round(np.exp(clf_ano.params),3)
            , 'Odds_Lim_inf': np.round(IC[:,0],3)
            , 'Odds_Lim_Sup': np.round(IC[:,1],3)
            , 'p-values':np.round(clf_ano.pvalues,3)
        }
    )
    print(tabulate(odds_ratio, headers = 'keys', tablefmt = 'grid'))

    with open(f'resultados/modelo2_V2_CNES/{periodo}_modelo0_{missing}_CNES_OBITO.txt', 'w') as f:
        f.write('---------------------------------------------------------------- \n')
        f.write('USING PROPENSITY SCORE TO SELECT/MATCH SAMPLES \n')
        f.write('----------------------------------------------------------------\n')
        f.write('N without missing  :' + str(po) +'\n')
        f.write('Treated samples    :' + str([tpo,np.round(100*tpo/po,2)]) + '% \n')
        f.write('Controled samples  :' + str([cpo,np.round(100*cpo/po,2)]) + '% \n')
        f.write('---------------------------------------------------------------- \n')
        f.write('SELECTED/MATCHET SAMPLES\n')
        f.write('---------------------------------------------------------------- \n')
        f.write('% Selected         :' + str(np.round(100*pa/po,2)) + '% \n')
        f.write('N selected         :' + str(pa) + '\n')
        f.write('Treated selected   :' + str([tpa,np.round(100*tpa/pa,2)]) + '% \n')
        f.write('Controled selected :' + str([cpa,np.round(100*cpa/pa,2)]) + '% \n')
        f.write('---------------------------------------------------------------- \n')
        f.write('PSW REPORT - ' + periodo+ '\n')
        f.write(str(clf_ano.summary(xname=aux+var_model)))
        f.write('\n')
        f.write(tabulate(odds_ratio, headers = 'keys', tablefmt = 'grid'))

    odds_ratio['periodo'] = periodo
    odds_ratio['modelo'] = f'modelo0_{missing}_CNES_CNES'
    odds_ratio.to_csv(f'resultados/modelo2_V2_CNES/{periodo}_modelo0_{missing}_CNES_OBITO.csv', decimal=',', sep=';', index=False)

    fig = sns.kdeplot(df_mod.query("ANO==0")["PROPENSITY_SCORE"],bw_adjust=.7, shade=False, color="r")
    fig = sns.kdeplot(df_mod.query("ANO==1")["PROPENSITY_SCORE"],bw_adjust=.7, shade=False, color="b")
    plt.legend(['Control','Treatment'])
    plt.savefig(f'resultados/modelo2_V2_CNES/fig1a_{periodo}_modelo0_{missing}_CNES_OBITO.png', format='png', dpi=300)
    plt.clf()
    #plt.show()
    #
    fig = sns.kdeplot(psw_base.query("ANO==0")["PROPENSITY_SCORE"],bw_adjust=0.7, shade=False, color="r")
    fig = sns.kdeplot(psw_base.query("ANO==1")["PROPENSITY_SCORE"],bw_adjust=0.7, shade=False, color="b")
    plt.legend(['Control','Treatment'])
    plt.savefig(f'resultados/modelo2_V2_CNES/fig1b_{periodo}_modelo0_{missing}_CNES_OBITO.png', format='png', dpi=300)
    plt.clf()
    #plt.show()
    #
    # #spearm
    var_corr = ['FLAG_BASE_SIM_DOFET'] + var_model

    plt.figure(figsize=(16, 6))
    # define the mask to set the values in the upper triangle to True

    mask = np.triu(np.ones_like(psw_base[var_corr].corr(method="spearman"), dtype=bool))
    heatmap = sns.heatmap(psw_base[var_corr].corr(method='spearman'), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Matrix', fontdict={'fontsize':18}, pad=16)
    plt.savefig(f'resultados/modelo2_V2_CNES/graf_corr_{periodo}_modelo0_{missing}_CNES_OBITO.png', format='png', dpi=300)
    plt.clf()
    # plt.show()
########################################################################################################################
    print("""
    Modelo 1
    Missing: Removido
    balanceamento: Antes do modelo 1
    Tipo de Balancemaneto: AAS undersampling
    """)
    # Separando df_mod
    df_mod = df_ano[variaveis_1]
    # Dummies
    df_mod = pd.get_dummies(df_mod)

    # AAS Sinasc
    # Separando as classes
    minority_class = df_mod[df_mod['FLAG_BASE_SIM_DOFET'] == 1]
    majority_class = df_mod[df_mod['FLAG_BASE_SIM_DOFET'] == 0]

    # Reduzindo a classe majoritária
    majority_class_sampled = majority_class.sample(n=len(minority_class), random_state=42)

    # Concatenando as duas classes
    df_mod = pd.concat([minority_class, majority_class_sampled]).reset_index(drop=True)
    # OBITO contagem
    NPT = (df_mod['FLAG_BASE_SIM_DOFET']==1).sum()
    NT = (df_mod['FLAG_BASE_SIM_DOFET']==0).sum()

    # Primeiro modelo
    #Defining covariates

    X = df_mod[var_model].values

    ## FIRST REGRESSION
    y = df_mod[['ANO']].values
    clf = LogisticRegression(random_state=0, max_iter=2000).fit(X, y)
    df_mod = df_mod.assign(PROPENSITY_SCORE=clf.predict_proba(X)[:, 1])

    #USING PROPENSITY SCORE TO SELECT SAMPLES TO SECOND REGRESSION
    psw_base=df_mod[((df_mod['PROPENSITY_SCORE']>df_mod['PROPENSITY_SCORE'].quantile(0.1)) &
                     (df_mod['PROPENSITY_SCORE']<df_mod['PROPENSITY_SCORE'].quantile(0.9)))]

    #COUNTING
    po=len(df_mod)
    tpo=len(df_mod[df_mod['ANO']==1])
    cpo=len(df_mod[df_mod['ANO']==0])
    pa=len(psw_base)
    tpa=len(psw_base[psw_base['ANO']==1])
    cpa=len(psw_base[psw_base['ANO']==0])

    print('----------------------------------------------------------------')
    print('USING PROPENSITY SCORE TO SELECT/MATCH SAMPLES')
    print('----------------------------------------------------------------')
    print('N without missing  :', po)
    print('Treated samples    :', tpo,np.round(100*tpo/po,2),'%')
    print('Controled samples  :', cpo,np.round(100*cpo/po,2),'%')
    print('----------------------------------------------------------------')
    print('SELECTED/MATCHET SAMPLES')
    print('----------------------------------------------------------------')
    print('% Selected         :', np.round(100*pa/po,2),'%')
    print('N selected         :', pa)
    print('Treated selected   :', tpa,np.round(100*tpa/pa,2),'%')
    print('Controled selected :', cpa,np.round(100*cpa/pa,2),'%')
    print('----------------------------------------------------------------')
    print('')

    ## SECOND REGRESSION

    aux = ['ANO']
    w = psw_base['ANO']/psw_base['PROPENSITY_SCORE'] + ((1-psw_base['ANO'])/(1-psw_base['PROPENSITY_SCORE']))

    X = psw_base[aux+var_model].astype(float).values
    y = psw_base[['FLAG_BASE_SIM_DOFET']].values

    X_ANO = sm.add_constant(X)
    clf_ano = sm.Logit(y, X_ANO, weights=w).fit(maxiter=1000)
    aux=['Intercept','ANO']

    print('----------------------------------------------------------------')
    print('PSW REPORT',periodo)
    print('----------------------------------------------------------------')
    print(clf_ano.summary(xname=aux+var_model))
    print('----------------------------------------------------------------')


    IC=np.exp(clf_ano.conf_int(0.05))
    odds_ratio=pd.DataFrame(
        data = {
            'Var':aux+var_model
            , 'Odds_ratio': np.round(np.exp(clf_ano.params),3)
            , 'Odds_Lim_inf': np.round(IC[:,0],3)
            , 'Odds_Lim_Sup': np.round(IC[:,1],3)
            , 'p-values':np.round(clf_ano.pvalues,3)
        }
    )
    print(tabulate(odds_ratio, headers = 'keys', tablefmt = 'grid'))

    with open(f'resultados/modelo2_V2_CNES/{periodo}_modelo1_{missing}_CNES_OBITO.txt', 'w') as f:
        f.write('---------------------------------------------------------------- \n')
        f.write('USING PROPENSITY SCORE TO SELECT/MATCH SAMPLES \n')
        f.write('----------------------------------------------------------------\n')
        f.write('N without missing  :' + str(po) +'\n')
        f.write('Treated samples    :' + str([tpo,np.round(100*tpo/po,2)]) + '% \n')
        f.write('Controled samples  :' + str([cpo,np.round(100*cpo/po,2)]) + '% \n')
        f.write('---------------------------------------------------------------- \n')
        f.write('SELECTED/MATCHET SAMPLES\n')
        f.write('---------------------------------------------------------------- \n')
        f.write('% Selected         :' + str(np.round(100*pa/po,2)) + '% \n')
        f.write('N selected         :' + str(pa) + '\n')
        f.write('Treated selected   :' + str([tpa,np.round(100*tpa/pa,2)]) + '% \n')
        f.write('Controled selected :' + str([cpa,np.round(100*cpa/pa,2)]) + '% \n')
        f.write('---------------------------------------------------------------- \n')
        f.write('PSW REPORT - ' + periodo+ '\n')
        f.write(str(clf_ano.summary(xname=aux+var_model)))
        f.write('\n')
        f.write(tabulate(odds_ratio, headers = 'keys', tablefmt = 'grid'))

    odds_ratio['periodo'] = periodo
    odds_ratio['modelo'] = f'modelo1_{missing}_CNES'
    odds_ratio.to_csv(f'resultados/modelo2_V2_CNES/{periodo}_modelo1_{missing}_CNES_OBITO.csv', decimal=',', sep=';', index=False)

    fig = sns.kdeplot(df_mod.query("ANO==0")["PROPENSITY_SCORE"], bw_adjust=.7, shade=False, color="r")
    fig = sns.kdeplot(df_mod.query("ANO==1")["PROPENSITY_SCORE"], bw_adjust=.7, shade=False, color="b")
    plt.legend(['Control', 'Treatment'])
    plt.savefig(f'resultados/modelo2_V2_CNES/fig1a_{periodo}_modelo1_{missing}_CNES_OBITO.png', format='png', dpi=300)
    plt.clf()
    # plt.show()
    #
    fig = sns.kdeplot(psw_base.query("ANO==0")["PROPENSITY_SCORE"], bw_adjust=0.7, shade=False, color="r")
    fig = sns.kdeplot(psw_base.query("ANO==1")["PROPENSITY_SCORE"], bw_adjust=0.7, shade=False, color="b")
    plt.legend(['Control', 'Treatment'])
    plt.savefig(f'resultados/modelo2_V2_CNES/fig1b_{periodo}_modelo1_{missing}_CNES_OBITO.png', format='png', dpi=300)
    plt.clf()
    # plt.show()
    #
    # #spearm
    var_corr = ['FLAG_BASE_SIM_DOFET'] + var_model

    plt.figure(figsize=(16, 6))
    # define the mask to set the values in the upper triangle to True

    mask = np.triu(np.ones_like(psw_base[var_corr].corr(method="spearman"), dtype=bool))
    heatmap = sns.heatmap(psw_base[var_corr].corr(method='spearman'), mask=mask, vmin=-1, vmax=1, annot=True,
                          cmap='BrBG')
    heatmap.set_title('Matrix', fontdict={'fontsize': 18}, pad=16)
    plt.savefig(f'resultados/modelo2_V2_CNES/graf_corr_{periodo}_modelo1_{missing}_CNES_OBITO.png', format='png', dpi=300)
    plt.clf()
    # plt.show()
########################################################################################################################
########################################################################################################################
########################################################################################################################
    print("""
    Modelo 2
    Missing: Removido
    balanceamento: Após o modelo 1
    Tipo de Balancemaneto: AAS undersampling
    """)
    df_mod = df_ano[variaveis_1]
    # Dummies
    df_mod = pd.get_dummies(df_mod)

    # OBITO contagem
    NPT = (df_mod['FLAG_BASE_SIM_DOFET']==1).sum()
    NT = (df_mod['FLAG_BASE_SIM_DOFET']==0).sum()

    # Primeiro modelo
    #Defining covariates
    X = df_mod[var_model].values

    ## FIRST REGRESSION
    y = df_mod[['ANO']].values
    clf = LogisticRegression(random_state=0, max_iter=2000).fit(X, y)
    df_mod = df_mod.assign(PROPENSITY_SCORE=clf.predict_proba(X)[:, 1])

    #USING PROPENSITY SCORE TO SELECT SAMPLES TO SECOND REGRESSION
    psw_base=df_mod[((df_mod['PROPENSITY_SCORE']>df_mod['PROPENSITY_SCORE'].quantile(0.1)) &
                     (df_mod['PROPENSITY_SCORE']<df_mod['PROPENSITY_SCORE'].quantile(0.9)))]

    #COUNTING
    po=len(df_mod)
    tpo=len(df_mod[df_mod['ANO']==1])
    cpo=len(df_mod[df_mod['ANO']==0])
    pa=len(psw_base)
    tpa=len(psw_base[psw_base['ANO']==1])
    cpa=len(psw_base[psw_base['ANO']==0])

    print('----------------------------------------------------------------')
    print('USING PROPENSITY SCORE TO SELECT/MATCH SAMPLES')
    print('----------------------------------------------------------------')
    print('N without missing  :', po)
    print('Treated samples    :', tpo,np.round(100*tpo/po,2),'%')
    print('Controled samples  :', cpo,np.round(100*cpo/po,2),'%')
    print('----------------------------------------------------------------')
    print('SELECTED/MATCHET SAMPLES')
    print('----------------------------------------------------------------')
    print('% Selected         :', np.round(100*pa/po,2),'%')
    print('N selected         :', pa)
    print('Treated selected   :', tpa,np.round(100*tpa/pa,2),'%')
    print('Controled selected :', cpa,np.round(100*cpa/pa,2),'%')
    print('----------------------------------------------------------------')
    print('')

    ## SECOND REGRESSION
    aux = ['ANO']
    psw_base['weights'] = psw_base['ANO']/psw_base['PROPENSITY_SCORE'] + ((1-psw_base['ANO'])/(1-psw_base['PROPENSITY_SCORE']))
    X = psw_base[aux+var_model].astype(float).values
    y = psw_base[['FLAG_BASE_SIM_DOFET']].values
    weights = psw_base['weights'].values

    # Aplicando Random UnderSampling para que a classe majoritária seja 50% maior que a minoritária
    #sampling_strategy = 1 / 1.5  # Aproximadamente 0.769
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)

    # Criando um DataFrame da base balanceada
    psw_base_balanced = pd.DataFrame(X_res, columns=aux + var_model)
    psw_base_balanced['FLAG_BASE_SIM_DOFET'] = y_res

    # Encontrando os índices das observações selecionadas
    selected_indices = rus.sample_indices_

    # Aplicando os pesos corretos às observações selecionadas
    weights_balanced = weights[selected_indices]
    psw_base_balanced['PROPENSITY_SCORE'] = psw_base['PROPENSITY_SCORE'].values[selected_indices]

    # Treinando o modelo final
    X_balanced = psw_base_balanced[aux + var_model].values
    y_balanced = psw_base_balanced[['FLAG_BASE_SIM_DOFET']].values

    X_ANO = sm.add_constant(X_balanced)
    clf_ano = sm.Logit(y_balanced, X_ANO, weights=weights_balanced).fit(maxiter=1000)

    aux=['Intercept','ANO']

    print('----------------------------------------------------------------')
    print('PSW REPORT',periodo)
    print('----------------------------------------------------------------')
    print(clf_ano.summary(xname=aux+var_model))
    print('----------------------------------------------------------------')

    IC=np.exp(clf_ano.conf_int(0.05))
    odds_ratio=pd.DataFrame(
        data = {
            'Var':aux+var_model
            , 'Odds_ratio': np.round(np.exp(clf_ano.params),3)
            , 'Odds_Lim_inf': np.round(IC[:,0],3)
            , 'Odds_Lim_Sup': np.round(IC[:,1],3)
            , 'p-values':np.round(clf_ano.pvalues,3)
        }
    )
    print(tabulate(odds_ratio, headers = 'keys', tablefmt = 'grid'))

    with open(f'resultados/modelo2_V2_CNES/{periodo}_modelo2_{missing}_CNES_OBITO.txt', 'w') as f:
        f.write('---------------------------------------------------------------- \n')
        f.write('USING PROPENSITY SCORE TO SELECT/MATCH SAMPLES \n')
        f.write('----------------------------------------------------------------\n')
        f.write('N without missing  :' + str(po) +'\n')
        f.write('Treated samples    :' + str([tpo,np.round(100*tpo/po,2)]) + '% \n')
        f.write('Controled samples  :' + str([cpo,np.round(100*cpo/po,2)]) + '% \n')
        f.write('---------------------------------------------------------------- \n')
        f.write('SELECTED/MATCHET SAMPLES\n')
        f.write('---------------------------------------------------------------- \n')
        f.write('% Selected         :' + str(np.round(100*pa/po,2)) + '% \n')
        f.write('N selected         :' + str(pa) + '\n')
        f.write('Treated selected   :' + str([tpa,np.round(100*tpa/pa,2)]) + '% \n')
        f.write('Controled selected :' + str([cpa,np.round(100*cpa/pa,2)]) + '% \n')
        f.write('---------------------------------------------------------------- \n')
        f.write('PSW REPORT - ' + periodo+ '\n')
        f.write(str(clf_ano.summary(xname=aux+var_model)))
        f.write('\n')
        f.write(tabulate(odds_ratio, headers = 'keys', tablefmt = 'grid'))

    odds_ratio['periodo'] = periodo
    odds_ratio['modelo'] = f'modelo2_{missing}_CNES'
    odds_ratio.to_csv(f'resultados/modelo2_V2_CNES/{periodo}_modelo2_{missing}_CNES_OBITO.csv', decimal=',', sep=';', index=False)

    fig = sns.kdeplot(df_mod.query("ANO==0")["PROPENSITY_SCORE"], bw_adjust=.7, shade=False, color="r")
    fig = sns.kdeplot(df_mod.query("ANO==1")["PROPENSITY_SCORE"], bw_adjust=.7, shade=False, color="b")
    plt.legend(['Control', 'Treatment'])
    plt.savefig(f'resultados/modelo2_V2_CNES/fig1a_{periodo}_modelo2_{missing}_CNES_OBITO.png', format='png', dpi=300)
    plt.clf()
    # plt.show()
    #
    fig = sns.kdeplot(psw_base_balanced.query("ANO==0")["PROPENSITY_SCORE"], bw_adjust=0.7, shade=False, color="r")
    fig = sns.kdeplot(psw_base_balanced.query("ANO==1")["PROPENSITY_SCORE"], bw_adjust=0.7, shade=False, color="b")
    plt.legend(['Control', 'Treatment'])
    plt.savefig(f'resultados/modelo2_V2_CNES/fig1b_{periodo}_modelo2_{missing}_CNES_OBITO.png', format='png', dpi=300)
    plt.clf()
    # plt.show()
    #
    # #spearm
    var_corr = ['FLAG_BASE_SIM_DOFET'] + var_model

    plt.figure(figsize=(16, 6))
    # define the mask to set the values in the upper triangle to True

    mask = np.triu(np.ones_like(psw_base_balanced[var_corr].corr(method="spearman"), dtype=bool))
    heatmap = sns.heatmap(psw_base_balanced[var_corr].corr(method='spearman'), mask=mask, vmin=-1, vmax=1, annot=True,
                          cmap='BrBG')
    heatmap.set_title('Matrix', fontdict={'fontsize': 18}, pad=16)
    plt.savefig(f'resultados/modelo2_V2_CNES/graf_corr_{periodo}_modelo2_{missing}_CNES_OBITO.png', format='png', dpi=300)
    plt.clf()
    # plt.show()
########################################################################################################################

    print("""
    Modelo 3
    Missing: Removido
    balanceamento: Após o modelo 1
    Tipo de Balancemaneto: AAS undersampling e tomek
    """)

    df_mod = df_ano[variaveis_1]

    # Dummies
    df_mod = pd.get_dummies(df_mod)

    # OBITO contagem
    NPT = (df_mod['FLAG_BASE_SIM_DOFET']==1).sum()
    NT = (df_mod['FLAG_BASE_SIM_DOFET']==0).sum()

    # Primeiro Modelo
    X = df_mod[var_model].values

    ## FIRST REGRESSION
    y = df_mod[['ANO']].values
    clf = LogisticRegression(random_state=0, max_iter=2000).fit(X, y)
    df_mod = df_mod.assign(PROPENSITY_SCORE=clf.predict_proba(X)[:, 1])

    #USING PROPENSITY SCORE TO SELECT SAMPLES TO SECOND REGRESSION
    psw_base=df_mod[((df_mod['PROPENSITY_SCORE']>df_mod['PROPENSITY_SCORE'].quantile(0.1)) &
                     (df_mod['PROPENSITY_SCORE']<df_mod['PROPENSITY_SCORE'].quantile(0.9)))]

    #COUNTING
    po=len(df_mod)
    tpo=len(df_mod[df_mod['ANO']==1])
    cpo=len(df_mod[df_mod['ANO']==0])
    pa=len(psw_base)
    tpa=len(psw_base[psw_base['ANO']==1])
    cpa=len(psw_base[psw_base['ANO']==0])

    print('----------------------------------------------------------------')
    print('USING PROPENSITY SCORE TO SELECT/MATCH SAMPLES')
    print('----------------------------------------------------------------')
    print('N without missing  :', po)
    print('Treated samples    :', tpo,np.round(100*tpo/po,2),'%')
    print('Controled samples  :', cpo,np.round(100*cpo/po,2),'%')
    print('----------------------------------------------------------------')
    print('SELECTED/MATCHET SAMPLES')
    print('----------------------------------------------------------------')
    print('% Selected         :', np.round(100*pa/po,2),'%')
    print('N selected         :', pa)
    print('Treated selected   :', tpa,np.round(100*tpa/pa,2),'%')
    print('Controled selected :', cpa,np.round(100*cpa/pa,2),'%')
    print('----------------------------------------------------------------')
    print('')

    ## SECOND REGRESSION

    aux = ['ANO']
    psw_base['weights'] = psw_base['ANO']/psw_base['PROPENSITY_SCORE'] + ((1-psw_base['ANO'])/(1-psw_base['PROPENSITY_SCORE']))
    # w = psw_base['ANO']/psw_base['PROPENSITY_SCORE'] + ((1-psw_base['ANO'])/(1-psw_base['PROPENSITY_SCORE']))

    X = psw_base[aux+var_model].astype(float).values
    y = psw_base[['FLAG_BASE_SIM_DOFET']].values
    weights = psw_base['weights'].values

    # Aplicando Random UnderSampling para que a classe majoritária seja 50% maior que a minoritária
    #sampling_strategy = 1 / 1.5  # Aproximadamente 0.769
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)

    # Aplicando Tomek Links
    tl = TomekLinks()
    X_res0, y_res0 = tl.fit_resample(X_res, y_res)

    # Criando um DataFrame da base balanceada
    psw_base_balanced = pd.DataFrame(X_res0, columns=aux + var_model)
    psw_base_balanced['FLAG_BASE_SIM_DOFET'] = y_res0

    # Encontrando os índices das observações selecionadas
    selected_indices = tl.sample_indices_

    # Aplicando os pesos corretos às observações selecionadas
    weights_balanced = weights[selected_indices]
    psw_base_balanced['PROPENSITY_SCORE'] = psw_base['PROPENSITY_SCORE'].values[selected_indices]

    # Treinando o modelo final
    X_balanced = psw_base_balanced[aux + var_model].values
    y_balanced = psw_base_balanced[['FLAG_BASE_SIM_DOFET']].values

    X_ANO = sm.add_constant(X_balanced)
    clf_ano = sm.Logit(y_balanced, X_ANO, weights=weights_balanced).fit(maxiter=1000)

    aux=['Intercept','ANO']

    print('----------------------------------------------------------------')
    print('PSW REPORT',periodo)
    print('----------------------------------------------------------------')
    print(clf_ano.summary(xname=aux+var_model))
    print('----------------------------------------------------------------')

    IC=np.exp(clf_ano.conf_int(0.05))
    odds_ratio=pd.DataFrame(
        data = {
            'Var':aux+var_model
            , 'Odds_ratio': np.round(np.exp(clf_ano.params),3)
            , 'Odds_Lim_inf': np.round(IC[:,0],3)
            , 'Odds_Lim_Sup': np.round(IC[:,1],3)
            , 'p-values':np.round(clf_ano.pvalues,3)
        }
    )
    print(tabulate(odds_ratio, headers = 'keys', tablefmt = 'grid'))

    with open(f'resultados/modelo2_V2_CNES/{periodo}_modelo3_{missing}_CNES_OBITO.txt', 'w') as f:
        f.write('---------------------------------------------------------------- \n')
        f.write('USING PROPENSITY SCORE TO SELECT/MATCH SAMPLES \n')
        f.write('----------------------------------------------------------------\n')
        f.write('N without missing  :' + str(po) +'\n')
        f.write('Treated samples    :' + str([tpo,np.round(100*tpo/po,2)]) + '% \n')
        f.write('Controled samples  :' + str([cpo,np.round(100*cpo/po,2)]) + '% \n')
        f.write('---------------------------------------------------------------- \n')
        f.write('SELECTED/MATCHET SAMPLES\n')
        f.write('---------------------------------------------------------------- \n')
        f.write('% Selected         :' + str(np.round(100*pa/po,2)) + '% \n')
        f.write('N selected         :' + str(pa) + '\n')
        f.write('Treated selected   :' + str([tpa,np.round(100*tpa/pa,2)]) + '% \n')
        f.write('Controled selected :' + str([cpa,np.round(100*cpa/pa,2)]) + '% \n')
        f.write('---------------------------------------------------------------- \n')
        f.write('PSW REPORT - ' + periodo+ '\n')
        f.write(str(clf_ano.summary(xname=aux+var_model)))
        f.write('\n')
        f.write(tabulate(odds_ratio, headers = 'keys', tablefmt = 'grid'))

    odds_ratio['periodo'] = periodo
    odds_ratio['modelo'] = f'modelo3_{missing}_CNES'
    odds_ratio.to_csv(f'resultados/modelo2_V2_CNES/{periodo}_modelo3_{missing}_CNES_OBITO.csv', decimal=',', sep=';',
                      index=False)

    fig = sns.kdeplot(df_mod.query("ANO==0")["PROPENSITY_SCORE"], bw_adjust=.7, shade=False, color="r")
    fig = sns.kdeplot(df_mod.query("ANO==1")["PROPENSITY_SCORE"], bw_adjust=.7, shade=False, color="b")
    plt.legend(['Control', 'Treatment'])
    plt.savefig(f'resultados/modelo2_V2_CNES/fig1a_{periodo}_modelo3_{missing}_CNES_OBITO.png', format='png', dpi=300)
    plt.clf()
    # plt.show()
    #
    fig = sns.kdeplot(psw_base_balanced.query("ANO==0")["PROPENSITY_SCORE"], bw_adjust=0.7, shade=False, color="r")
    fig = sns.kdeplot(psw_base_balanced.query("ANO==1")["PROPENSITY_SCORE"], bw_adjust=0.7, shade=False, color="b")
    plt.legend(['Control', 'Treatment'])
    plt.savefig(f'resultados/modelo2_V2_CNES/fig1b_{periodo}_modelo3_{missing}_CNES_OBITO.png', format='png', dpi=300)
    plt.clf()
    # plt.show()
    #
    # #spearm
    var_corr = ['FLAG_BASE_SIM_DOFET'] + var_model

    plt.figure(figsize=(16, 6))
    # define the mask to set the values in the upper triangle to True

    mask = np.triu(np.ones_like(psw_base_balanced[var_corr].corr(method="spearman"), dtype=bool))
    heatmap = sns.heatmap(psw_base_balanced[var_corr].corr(method='spearman'), mask=mask, vmin=-1, vmax=1, annot=True,
                          cmap='BrBG')
    heatmap.set_title('Matrix', fontdict={'fontsize': 18}, pad=16)
    plt.savefig(f'resultados/modelo2_V2_CNES/graf_corr_{periodo}_modelo3_{missing}_CNES_OBITO.png', format='png', dpi=300)
    plt.clf()
    # plt.show()