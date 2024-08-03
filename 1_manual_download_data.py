import os
import re
import shutil
from Download_data.import_data_pcdas import *

"""
# Importing the data manually
Links:
https://bigdata-arquivos.icict.fiocruz.br/PUBLICO/SIM_DOFET/ETLSIM.DOFET.zip
https://bigdata-arquivos.icict.fiocruz.br/PUBLICO/SINASC/ETLSINASC.zip
https://bigdata-arquivos.icict.fiocruz.br/PUBLICO/CNES/ETLCNES.zip

Save to the folder, unzip and run the codes below
"""

#------------------------------------------ SINASC ------------------------------------------------------------------
# Give the path where the files were saved and where you want to save the years of interest
############################################################################################
pasta_origem = r"D:/ETLSINASC"
pasta_destino = r"C:/Users/gabri/Documents/PSW_COVID_OBITO_FETAL"
year =[2018, 2019, 2020, 2021, 2022]
#############################################################################################
PCDAS = Import_Data_PCDAS()
# Identify the files in the folder
lista_arq = PCDAS.function_list_path_files(path=pasta_origem)
# Filter addresses according to the years of interest
enderecos_filtrados = [endereco for endereco in lista_arq if any(str(ano) in endereco for ano in year)]
# Creating the folder in the path
nova_pasta = f'{pasta_destino}/SINASC'
PCDAS.function_create_path(path_download=nova_pasta)
# Copying the files to the new folder
for i in enderecos_filtrados:
    shutil.copy(i, nova_pasta)

#------------------------------------------ CNES ------------------------------------------------------------------
# Give the path where the files were saved and where you want to save the years of interest
############################################################################################
pasta_origem = r"D:/ETLCNES"
pasta_destino = r"C:/Users/gabri/Documents/PSW_COVID_OBITO_FETAL/"
year = [2018, 2019, 2020, 2021, 2022]
#############################################################################################
PCDAS = Import_Data_PCDAS()
lista_arq = PCDAS.function_list_path_files(path=pasta_origem)
year = [f'__{str(i)[2:]}_' for i in year]
# Filter addresses according to the years of interest
enderecos_filtrados = [endereco for endereco in lista_arq if any(str(ano) in endereco for ano in year)]
# Creating the folder in the path
nova_pasta = f'{pasta_destino}/CNES'
PCDAS.function_create_path(path_download=nova_pasta)
# Copying the files to the new folder
for i in enderecos_filtrados:
    shutil.copy(i, nova_pasta)

#------------------------------------------ SIM_DOFET ------------------------------------------------------------------
# Give the path where the files were saved and where you want to save the years of interest
###########################################################################################
pasta_origem = r"D:/ETLSIM.DOFET/dados/apache-airflow/data/SIM_DOFET"
pasta_destino = r"C:/Users/gabri/Documents/PSW_COVID_OBITO_FETAL"
year =[2018, 2019, 2020, 2021, 2022]
###########################################################################################
PCDAS = Import_Data_PCDAS()
# Identify the files in the folder
lista_arq = PCDAS.function_list_path_files(path=pasta_origem)
# Filter addresses according to the years of interest
enderecos_filtrados = [endereco for endereco in lista_arq if any(str(ano) in endereco for ano in year)]
# Creating the folder in the path
nova_pasta = f'{pasta_destino}/SIM_DOFET'
PCDAS.function_create_path(path_download=nova_pasta)
# Copying the files to the new folder
for i in enderecos_filtrados:
    shutil.copy(i, nova_pasta)