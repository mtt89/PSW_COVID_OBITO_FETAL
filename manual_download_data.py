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
year =[2019, 2020, 2021, 2022]
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
year = [2019, 2020, 2021, 2022]
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
year =[2019, 2020, 2021, 2022]
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






# Create a regular expression to find the year in the file name
regex_ano = re.compile(r"ETLSIM\.DOFET_(\d{4})_t")

# Cycle through all files in the source folder
for arquivo in os.listdir(pasta_origem):
    caminho_completo = os.path.join(pasta_origem, arquivo)

    # Check if it is a file
    if os.path.isfile(caminho_completo):
        # Extract the year from the file name
        match = regex_ano.match(arquivo)
        if match:
            ano = match.group(1)
            # Checks if the year is between ano_inicio and ano_fim

            if ano_inicio <= int(ano) <= ano_fim:
                # Checks if the year is between 2019 and 2022
                pasta_destino_ano = os.path.join(pasta_destino, f"ANO_{ano}")
                # Create destination folder if it doesn't already exist
                os.makedirs(pasta_destino_ano, exist_ok=True)
                # Moves the file to the corresponding destination folder
                shutil.copy(caminho_completo, pasta_destino_ano)

