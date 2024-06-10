import pandas as pd
import requests
import os
import re
import zipfile
import shutil

class Import_Data_PCDAS:
    """
    Class to download public health data from the PCDAS platform.
    Attributes:
        dataset: The name of the dataset.
    """
    def __init__(self, dataset: str):
        self.dataset = 'SIM_DOFET'


    @staticmethod
    def download_data(url: str, path_download: str):
        """
        Downloads a file from a URL and saves it to the specified location.

        Args:
          url (str): The URL of the file you want to download.
          path_download (str): The full path of the location where you want to save the file.

        Returns:
          bool: `True` if the download was successful, `False` otherwise.
        """
        name_arq = url.split('/')
        name_arq = name_arq[(len(name_arq) - 1)]
        path_download = f'{path_download}/{name_arq}'
        try:
            resposta = requests.get(url)
            print(f'Request status {resposta.status_code}')
            with open(path_download, 'wb') as arquivo:
                for chunk in resposta.iter_content():
                    arquivo.write(chunk)

            # Checks if the downloaded file is a ZIP file
            if zipfile.is_zipfile(path_download):
                # Sets the extraction directory to the same folder where the ZIP file was downloaded
                extracao_dir = os.path.splitext(path_download)[0]
                # Extract the contents of the ZIP file to the same directory
                with zipfile.ZipFile(path_download, 'r') as zip_ref:
                    zip_ref.extractall(extracao_dir)
                # Remove ZIP file after unzipping
                os.remove(path_download)
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error when downloading file {url}: {e}")
            return False

    @staticmethod
    def create_path(path_download: str):
        """
        Creates a folder in a directory if it does not exist
        :param path_download(str): The full path of the location where you want to create the folder.
        :return:
        """
        # Verifica se o diretório não existe
        if not os.path.exists(path_download):
            try:
                # Cria o diretório
                os.makedirs(path_download)
                print(f"Pasta '{path_download}' criada com sucesso!")
            except OSError as e:
                # Em caso de erro
                print(f"Erro ao criar pasta '{path_download}': {e}")
        else:
            print(f"Pasta '{path_download}' já existe.")
    @staticmethod
    def list_path_files(path: str):
        """
        Maps files to a specified path
        :param path (str): Path of the folder that should be mapped
        :return: Returns a list with the path of all files in the folder.
        """
        arquivos = []
        for diretorio_raiz, _, arquivos_na_pasta in os.walk(path):
            for arquivo in arquivos_na_pasta:
                endereco_completo = os.path.join(diretorio_raiz, arquivo)
                arquivos.append(endereco_completo)
        return arquivos
########################################################################################################################

    def function_import_sim_dofet(self, path_download: str, year: list):
        url = 'https://bigdata-arquivos.icict.fiocruz.br/PUBLICO/SIM_DOFET/ETLSIM.DOFET.zip'
        aa = self.download_data(url=url, path_download=path_download)
        if aa:
            # Identique os arquivos na pasta
            path_ = f'{path_download}/ETLSIM.DOFET/dados/apache-airflow/data/SIM_DOFET'
            lista_arq = self.list_path_files(path=path_)
            # Filtrar endereços de acordo com os anos de interesse
            enderecos_filtrados = [endereco for endereco in lista_arq if any(str(ano) in endereco for ano in year)]
            # Criando a pasta no caminho
            nova_pasta = f'{path_download}/SIM_DOFET'
            self.create_path(path_download=nova_pasta)
            # Copiando os arquivos para a nova pasta
            for i in enderecos_filtrados:
                shutil.copy(i, nova_pasta)
            # Apaga a pasta que foi baixada
            shutil.rmtree(f'{path_download}/ETLSIM.DOFET')
            print('Pasta SIM_DOFET criada com sucesso')
        else:
            print(f'Verifique a url {url} ou o caminho {path_download}')
        pass
