# Importing Modules
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from tabulate import tabulate
import warnings
from Funcoes_auxiliares.func_aux import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

warnings.filterwarnings("ignore")

df = pd.read_csv('base_limpa/base_unificada_limpa_com_input.csv')
# Peso calculado
df['cat_peso_calc'] = [
    func_peso_calculado(sexo, peso, int(round(semana_gest, 0))) for sexo, peso, semana_gest in
    zip(df['SEXO'], df['PESO'], df['SEMAGESTAC'])
]

# Função para filtrar o dataframe e adicionar a coluna 'ANO'
def filtrar_e_adicionar(df, per):
    anos = list(map(int, per.split('_')))
    df_ano = df.loc[df['ano_evento'].isin(anos)]
    df_ano = df_ano.reset_index(drop=True)
    df_ano['ANO'] = [0 if i == anos[0] else 1 for i in df_ano['ano_evento']]
    return df_ano


lista_periodo = [
    '2018_2019', '2018_2020', '2018_2021', '2018_2022', '2019_2020', '2019_2021', '2019_2022', '2020_2021', '2020_2022'
    , '2021_2022', '2018_2019_2020', '2018_2019_2021', '2018_2019_2022', '2018_2020_2021', '2018_2020_2022',
    '2018_2021_2022'
    , '2019_2020_2021', '2019_2020_2022', '2019_2021_2022', '2020_2021_2022', '2018_2019_2020_2021',
    '2018_2019_2020_2022'
    , '2018_2019_2021_2022', '2018_2020_2021_2022', '2019_2020_2021_2022', '2018_2019_2020_2021_2022'
]

# Exemplos de uso
contador = 0
for periodo in lista_periodo:
    print(f'{periodo} - rodada {contador} de {len(lista_periodo)}')
    missing = 'missing_com_input'
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
       # , 'idade_gestacao_faixa'
        , 'def_sexo'
        #, 'peso_faixa'
        , 'cat_peso_calc'
        , 'FLAG_BASE'
    ]
    df_mod = df_ano[variaveis_1]

    # Dummies
    df_mod = pd.get_dummies(df_mod)

    # OBITO contagem
    NPT = (df_mod['FLAG_BASE_SIM_DOFET'] == 1).sum()
    NT = (df_mod['FLAG_BASE_SIM_DOFET'] == 0).sum()
    # Defining covariates
    var_model = [
        'evento_REGIAO_Centro-Oeste'
        , 'evento_REGIAO_Nordeste'
        , 'evento_REGIAO_Norte'
        # , 'evento_REGIAO_Sudeste'
        , 'evento_REGIAO_Sul'
        # , 'idademae_faixa_entre_20_34'
        , 'idademae_faixa_entre_35_39'
        , 'idademae_faixa_maior_igual_40'
        , 'idademae_faixa_menor_igual_19'
        , 'escolaridade_mae_Ensino_medio'
        # , 'escolaridade_mae_Ensino_superior'
        , 'escolaridade_mae_Fundamental'
        , 'escolaridade_mae_Sem_escolaridade'
        , 'tipo_gravidez_Multipla'
        # , 'tipo_gravidez_Unica'
        # , 'PRM'
        # , 'def_sexo_Feminino'
        , 'def_sexo_Masculino'
        # , 'cat_peso_calc_AIG'
        , 'cat_peso_calc_GIG'
        , 'cat_peso_calc_PIG'
        # , 'peso_faixa_entre_1500_2499'
        # , 'peso_faixa_entre_2500_3500'
        # , 'peso_faixa_entre_3500_3999'
        # , 'peso_faixa_entre_500_1499'
        # , 'peso_faixa_maior_igual_4000'
        # , 'peso_faixa_menor_500'
        # , 'idade_gestacao_faixa_entre_22_27'
        # , 'idade_gestacao_faixa_entre_28_36'
        # , 'idade_gestacao_faixa_entre_40_42'
        # # ,  'idade_gestacao_faixa_entre_37_39'
        # ,'cat_CENTROBS'
        # , 'cat_QTINST34'
        # ,'cat_QTINST35'
        # ,'cat_QTINST36'
        # , 'cat_QTINST37'
        # , 'cat_QTLEIT34'
        # , 'cat_QTLEIT38'
        # , 'cat_QTLEIT39'
        # , 'cat_QTLEIT40'
        # , 'cat_CENTRNEO'
        # , 'cat_TP_UNID_5'
        # , 'cat_TP_UNID_7'
        # , 'cat_TP_UNID_15'
        # , 'cat_TP_UNID_36'
    ]
    ########################################################################################################################
    # ---------------------------------       Modelo Completo         --------------------------------------
    ########################################################################################################################
    print('---------------------------------       Modelo Completo         --------------------------------------')
    print("""
        Modelo 0 - completo
        Missing: input
        balanceamento: sem
        Tipo de Balancemaneto: sem
        """)

    # Primeiro modelo
    # Defining covariates

    X = df_mod[var_model].values

    ## FIRST REGRESSION
    y = df_mod[['ANO']].values
    clf = LogisticRegression(random_state=0, max_iter=2000).fit(X, y)
    df_mod = df_mod.assign(PROPENSITY_SCORE=clf.predict_proba(X)[:, 1])

    # USING PROPENSITY SCORE TO SELECT SAMPLES TO SECOND REGRESSION
    psw_base = df_mod[((df_mod['PROPENSITY_SCORE'] > df_mod['PROPENSITY_SCORE'].quantile(0.1)) &
                       (df_mod['PROPENSITY_SCORE'] < df_mod['PROPENSITY_SCORE'].quantile(0.9)))]

    # COUNTING
    po = len(df_mod)
    tpo = len(df_mod[df_mod['ANO'] == 1])
    cpo = len(df_mod[df_mod['ANO'] == 0])
    pa = len(psw_base)
    tpa = len(psw_base[psw_base['ANO'] == 1])
    cpa = len(psw_base[psw_base['ANO'] == 0])
    #
    # print('----------------------------------------------------------------')
    # print('USING PROPENSITY SCORE TO SELECT/MATCH SAMPLES')
    # print('----------------------------------------------------------------')
    # print('N without missing  :', po)
    # print('Treated samples    :', tpo, np.round(100 * tpo / po, 2), '%')
    # print('Controled samples  :', cpo, np.round(100 * cpo / po, 2), '%')
    # print('----------------------------------------------------------------')
    # print('SELECTED/MATCHET SAMPLES')
    # print('----------------------------------------------------------------')
    # print('% Selected         :', np.round(100 * pa / po, 2), '%')
    # print('N selected         :', pa)
    # print('Treated selected   :', tpa, np.round(100 * tpa / pa, 2), '%')
    # print('Controled selected :', cpa, np.round(100 * cpa / pa, 2), '%')
    # print('----------------------------------------------------------------')
    # print('')

    ## SECOND REGRESSION

    aux = ['ANO']
    w = psw_base['ANO'] / psw_base['PROPENSITY_SCORE'] + ((1 - psw_base['ANO']) / (1 - psw_base['PROPENSITY_SCORE']))

    X = psw_base[aux + var_model].astype(float).values
    y = psw_base[['FLAG_BASE_SIM_DOFET']].values

    X_ANO = sm.add_constant(X)
    clf_ano = sm.Logit(y, X_ANO, weights=w).fit(maxiter=1000)
    aux = ['Intercept', 'ANO']

    # print('----------------------------------------------------------------')
    # print('PSW REPORT', periodo)
    # print('----------------------------------------------------------------')
    # print(clf_ano.summary(xname=aux + var_model))
    # print('----------------------------------------------------------------')

    IC = np.exp(clf_ano.conf_int(0.05))
    odds_ratio = pd.DataFrame(
        data={
            'Var': aux + var_model
            , 'Odds_ratio': np.round(np.exp(clf_ano.params), 3)
            , 'Odds_Lim_inf': np.round(IC[:, 0], 3)
            , 'Odds_Lim_Sup': np.round(IC[:, 1], 3)
            , 'p-values': np.round(clf_ano.pvalues, 3)
        }
    )
    # print(tabulate(odds_ratio, headers='keys', tablefmt='grid'))

    with open(f'resultados/modelo_3_sem_subgrupo/{periodo}_modelo0_{missing}_OBITO.txt', 'w') as f:
        f.write('---------------------------------------------------------------- \n')
        f.write('USING PROPENSITY SCORE TO SELECT/MATCH SAMPLES \n')
        f.write('----------------------------------------------------------------\n')
        f.write('N without missing  :' + str(po) + '\n')
        f.write('Treated samples    :' + str([tpo, np.round(100 * tpo / po, 2)]) + '% \n')
        f.write('Controled samples  :' + str([cpo, np.round(100 * cpo / po, 2)]) + '% \n')
        f.write('---------------------------------------------------------------- \n')
        f.write('SELECTED/MATCHET SAMPLES\n')
        f.write('---------------------------------------------------------------- \n')
        f.write('% Selected         :' + str(np.round(100 * pa / po, 2)) + '% \n')
        f.write('N selected         :' + str(pa) + '\n')
        f.write('Treated selected   :' + str([tpa, np.round(100 * tpa / pa, 2)]) + '% \n')
        f.write('Controled selected :' + str([cpa, np.round(100 * cpa / pa, 2)]) + '% \n')
        f.write('---------------------------------------------------------------- \n')
        f.write('PSW REPORT - ' + periodo + '\n')
        f.write(str(clf_ano.summary(xname=aux + var_model)))
        f.write('\n')
        f.write(tabulate(odds_ratio, headers='keys', tablefmt='grid'))

    odds_ratio['periodo'] = periodo
    odds_ratio['modelo'] = f'modelo0_{missing}'
    odds_ratio.to_csv(f'resultados/modelo_3_sem_subgrupo/{periodo}_modelo0_{missing}_OBITO.csv', decimal=',', sep=';',
                      index=False)

    fig = sns.kdeplot(df_mod.query("ANO==0")["PROPENSITY_SCORE"], bw_adjust=.7, shade=False, color="r")
    fig = sns.kdeplot(df_mod.query("ANO==1")["PROPENSITY_SCORE"], bw_adjust=.7, shade=False, color="b")
    plt.legend(['Control', 'Treatment'])
    plt.savefig(f'resultados/modelo_3_sem_subgrupo/fig1a_{periodo}_modelo0_{missing}_OBITO.png', format='png', dpi=300)
    plt.clf()
    # plt.show()
    #
    fig = sns.kdeplot(psw_base.query("ANO==0")["PROPENSITY_SCORE"], bw_adjust=0.7, shade=False, color="r")
    fig = sns.kdeplot(psw_base.query("ANO==1")["PROPENSITY_SCORE"], bw_adjust=0.7, shade=False, color="b")
    plt.legend(['Control', 'Treatment'])
    plt.savefig(f'resultados/modelo_3_sem_subgrupo/fig1b_{periodo}_modelo0_{missing}_OBITO.png', format='png', dpi=300)
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
    plt.savefig(f'resultados/modelo_3_sem_subgrupo/graf_corr_{periodo}_modelo0_{missing}_OBITO.png', format='png',
                dpi=300)
    plt.clf()
    contador+=1
    # plt.show()
