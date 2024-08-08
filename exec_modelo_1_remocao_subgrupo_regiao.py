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

df = pd.read_csv('base_limpa/base_unificada_limpa_remocao.csv')
# Peso calculado
df['cat_peso_calc'] = [
    func_peso_calculado(sexo, peso, int(round(semana_gest, 0))) for sexo, peso, semana_gest in
    zip(df['SEXO'], df['PESO'], df['SEMAGESTAC'])
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
    '2018_2019', '2018_2020', '2018_2021', '2018_2022', '2019_2020', '2019_2021', '2019_2022', '2020_2021', '2020_2022'
    , '2021_2022', '2018_2019_2020', '2018_2019_2021', '2018_2019_2022', '2018_2020_2021', '2018_2020_2022','2018_2021_2022'
    , '2019_2020_2021', '2019_2020_2022', '2019_2021_2022', '2020_2021_2022', '2018_2019_2020_2021', '2018_2019_2020_2022'
    , '2018_2019_2021_2022', '2018_2020_2021_2022', '2019_2020_2021_2022', '2018_2019_2020_2021_2022'
]

lista_regiao = ['Sudeste', 'Norte', 'Nordeste', 'Sul', 'Centro-Oeste']
for regiao in lista_regiao:
    df_regiao = df[df['evento_REGIAO'] == regiao]
    df_regiao = df_regiao.reset_index(drop=True)
    for periodo in lista_periodo:
        missing = 'missing_removido'  # 'missing_com_input'
        df_ano = filtrar_e_adicionar(df_regiao, periodo)
        df_ano['ano_evento'].value_counts()
        df_ano['cat_peso_calc'].value_counts()

        # Separando df_mod
        variaveis_1 = [
            'ANO'
            #, 'evento_REGIAO'
            , 'idademae_faixa'
            , 'escolaridade_mae'
            , 'tipo_gravidez'
            , 'idade_gestacao_faixa'
            , 'def_sexo'
            , 'peso_faixa'
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
            # 'evento_REGIAO_Centro-Oeste'
            # , 'evento_REGIAO_Nordeste'
            # , 'evento_REGIAO_Norte'
            # # , 'evento_REGIAO_Sudeste'
            # , 'evento_REGIAO_Sul'
            # , 'idademae_faixa_entre_20_34'
             'idademae_faixa_entre_35_39'
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
        print("""
        Modelo 1 - avaliando overfitting
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
        NPT = (df_mod['FLAG_BASE_SIM_DOFET'] == 1).sum()
        NT = (df_mod['FLAG_BASE_SIM_DOFET'] == 0).sum()

        # Primeiro modelo
        # Defining covariates

        ## FIRST REGRESSION
        X = df_mod[var_model].values
        y = df_mod[['ANO']].values

        # Dividir os dados em conjuntos de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        clf = LogisticRegression(random_state=0, max_iter=2000).fit(X_train, y_train)
        ####################################################################################################################
        # Testando overfitting
        # Prever no conjunto de teste
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        y_pred_proba_train = clf.predict_proba(X_train)[:, 1]
        y_pred_proba_test = clf.predict_proba(X_test)[:, 1]

        # Avaliar o modelo
        metrics_first_model = {
            'ROC AUC Train': roc_auc_score(y_train, y_pred_proba_train),
            'Accuracy Train': accuracy_score(y_train, y_pred_train),
            'Precision Train': precision_score(y_train, y_pred_train),
            'Recall Train': recall_score(y_train, y_pred_train),
            'ROC AUC Test': roc_auc_score(y_test, y_pred_proba_test),
            'Accuracy Test': accuracy_score(y_test, y_pred_test),
            'Precision Test': precision_score(y_test, y_pred_test),
            'Recall Test': recall_score(y_test, y_pred_test)
        }
        ####################################################################################################################
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

        print('----------------------------------------------------------------')
        print('USING PROPENSITY SCORE TO SELECT/MATCH SAMPLES')
        print('----------------------------------------------------------------')
        print('N without missing  :', po)
        print('Treated samples    :', tpo, np.round(100 * tpo / po, 2), '%')
        print('Controled samples  :', cpo, np.round(100 * cpo / po, 2), '%')
        print('----------------------------------------------------------------')
        print('SELECTED/MATCHET SAMPLES')
        print('----------------------------------------------------------------')
        print('% Selected         :', np.round(100 * pa / po, 2), '%')
        print('N selected         :', pa)
        print('Treated selected   :', tpa, np.round(100 * tpa / pa, 2), '%')
        print('Controled selected :', cpa, np.round(100 * cpa / pa, 2), '%')
        print('----------------------------------------------------------------')
        print('')

        ## SECOND REGRESSION

        aux = ['ANO']
        w = psw_base['ANO'] / psw_base['PROPENSITY_SCORE'] + ((1 - psw_base['ANO']) / (1 - psw_base['PROPENSITY_SCORE']))
        ###########################################################################
        # Dividir os dados da segunda regressão em treino e teste
        X_second = psw_base[['ANO'] + var_model].astype(float).values
        y_second = psw_base[['FLAG_BASE_SIM_DOFET']].values
        X_train_second, X_test_second, y_train_second, y_test_second, w_train, w_test = train_test_split(
            X_second, y_second, w, test_size=0.1, random_state=0)

        # Adicionar constante
        X_train_second_ANO = sm.add_constant(X_train_second)
        X_test_second_ANO = sm.add_constant(X_test_second)

        # Treinar o modelo na parte de treino
        clf_ano_train = sm.Logit(y_train_second, X_train_second_ANO, weights=w_train).fit(maxiter=1000)
        aux = ['Intercept', 'ANO']

        # Prever na parte de treino
        y_pred_proba_train_ano = clf_ano_train.predict(X_train_second_ANO)
        y_pred_train_ano = (y_pred_proba_train_ano >= 0.5).astype(int)

        # Prever na parte de teste
        y_pred_proba_test_ano = clf_ano_train.predict(X_test_second_ANO)
        y_pred_test_ano = (y_pred_proba_test_ano >= 0.5).astype(int)

        # Avaliar o modelo
        metrics_second_model = {
            'ROC AUC Train': roc_auc_score(y_train_second, y_pred_proba_train_ano),
            'Accuracy Train': accuracy_score(y_train_second, y_pred_train_ano),
            'Precision Train': precision_score(y_train_second, y_pred_train_ano),
            'Recall Train': recall_score(y_train_second, y_pred_train_ano),
            'ROC AUC Test': roc_auc_score(y_test_second, y_pred_proba_test_ano),
            'Accuracy Test': accuracy_score(y_test_second, y_pred_test_ano),
            'Precision Test': precision_score(y_test_second, y_pred_test_ano),
            'Recall Test': recall_score(y_test_second, y_pred_test_ano)
        }
        # Salvar as métricas em um arquivo CSV
        df_metrics_first_model = pd.DataFrame([metrics_first_model])
        df_metrics_second_model = pd.DataFrame([metrics_second_model])

        # Adicionar identificadores de modelo para referência
        df_metrics_first_model['Model'] = 'First Model (Propensity Score)'
        df_metrics_second_model['Model'] = 'Second Model (Outcome)'

        # Combinar os DataFrames
        df_metrics = pd.concat([df_metrics_first_model, df_metrics_second_model],
                               ignore_index=True)
        df_metrics['periodo'] = periodo
        df_metrics['modelo'] = f'modelo1_{missing}'
        df_metrics['regiao'] = regiao
        # Salvar em um arquivo CSV
        df_metrics.to_csv(f'resultados/modelo_3_subgrupo_regiao/{periodo}_modelo1_{missing}_{regiao}_teste_overfitting.csv', decimal=',', sep=';',
                          index=False)

        print('----------------------------------------------------------------')
        print('PSW REPORT overfitting', periodo)
        print('----------------------------------------------------------------')
        print(clf_ano_train.summary(xname=aux + var_model))
        print('----------------------------------------------------------------')

        IC = np.exp(clf_ano_train.conf_int(0.05))
        odds_ratio = pd.DataFrame(
            data={
                'Var': aux + var_model
                , 'Odds_ratio': np.round(np.exp(clf_ano_train.params), 3)
                , 'Odds_Lim_inf': np.round(IC[:, 0], 3)
                , 'Odds_Lim_Sup': np.round(IC[:, 1], 3)
                , 'p-values': np.round(clf_ano_train.pvalues, 3)
            }
        )
        print(tabulate(odds_ratio, headers='keys', tablefmt='grid'))
        ########################################################################################################################
        # ---------------------------------       Modelo Completo         --------------------------------------
        ########################################################################################################################
        print('---------------------------------       Modelo Completo         --------------------------------------')
        print("""
            Modelo 1 - completo
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
        NPT = (df_mod['FLAG_BASE_SIM_DOFET'] == 1).sum()
        NT = (df_mod['FLAG_BASE_SIM_DOFET'] == 0).sum()

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

        print('----------------------------------------------------------------')
        print('USING PROPENSITY SCORE TO SELECT/MATCH SAMPLES')
        print('----------------------------------------------------------------')
        print('N without missing  :', po)
        print('Treated samples    :', tpo, np.round(100 * tpo / po, 2), '%')
        print('Controled samples  :', cpo, np.round(100 * cpo / po, 2), '%')
        print('----------------------------------------------------------------')
        print('SELECTED/MATCHET SAMPLES')
        print('----------------------------------------------------------------')
        print('% Selected         :', np.round(100 * pa / po, 2), '%')
        print('N selected         :', pa)
        print('Treated selected   :', tpa, np.round(100 * tpa / pa, 2), '%')
        print('Controled selected :', cpa, np.round(100 * cpa / pa, 2), '%')
        print('----------------------------------------------------------------')
        print('')

        ## SECOND REGRESSION

        aux = ['ANO']
        w = psw_base['ANO'] / psw_base['PROPENSITY_SCORE'] + ((1 - psw_base['ANO']) / (1 - psw_base['PROPENSITY_SCORE']))

        X = psw_base[aux + var_model].astype(float).values
        y = psw_base[['FLAG_BASE_SIM_DOFET']].values

        X_ANO = sm.add_constant(X)
        clf_ano = sm.Logit(y, X_ANO, weights=w).fit(maxiter=1000)
        aux = ['Intercept', 'ANO']

        print('----------------------------------------------------------------')
        print('PSW REPORT', periodo)
        print('----------------------------------------------------------------')
        print(clf_ano.summary(xname=aux + var_model))
        print('----------------------------------------------------------------')

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
        print(tabulate(odds_ratio, headers='keys', tablefmt='grid'))

        with open(f'resultados/modelo_3_subgrupo_regiao/{periodo}_modelo1_{missing}_{regiao}_OBITO.txt', 'w') as f:
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
        odds_ratio['modelo'] = f'modelo1_{missing}'
        odds_ratio['regiao'] = regiao
        odds_ratio.to_csv(f'resultados/modelo_3_subgrupo_regiao/{periodo}_modelo1_{missing}_{regiao}_OBITO.csv', decimal=',', sep=';',
                          index=False)

        fig = sns.kdeplot(df_mod.query("ANO==0")["PROPENSITY_SCORE"], bw_adjust=.7, shade=False, color="r")
        fig = sns.kdeplot(df_mod.query("ANO==1")["PROPENSITY_SCORE"], bw_adjust=.7, shade=False, color="b")
        plt.legend(['Control', 'Treatment'])
        plt.savefig(f'resultados/modelo_3_subgrupo_regiao/fig1a_{periodo}_modelo1_{missing}_{regiao}_OBITO.png', format='png', dpi=300)
        plt.clf()
        # plt.show()
        #
        fig = sns.kdeplot(psw_base.query("ANO==0")["PROPENSITY_SCORE"], bw_adjust=0.7, shade=False, color="r")
        fig = sns.kdeplot(psw_base.query("ANO==1")["PROPENSITY_SCORE"], bw_adjust=0.7, shade=False, color="b")
        plt.legend(['Control', 'Treatment'])
        plt.savefig(f'resultados/modelo_3_subgrupo_regiao/fig1b_{periodo}_modelo1_{missing}_{regiao}_OBITO.png', format='png', dpi=300)
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
        plt.savefig(f'resultados/modelo_3_subgrupo_regiao/graf_corr_{periodo}_modelo1_{missing}_{regiao}_OBITO.png', format='png', dpi=300)
        plt.clf()
        # plt.show()
