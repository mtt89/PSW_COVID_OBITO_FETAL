import pandas as pd
import requests
import os
import re
import zipfile
import shutil

class Import_Data_PCDAS:
    """
    Class to download public health data from the PCDAS platform.
    """
    @staticmethod
    def function_download_data(url: str, path_download: str):
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
    def function_create_path(path_download: str):
        """
        Creates a folder in a directory if it does not exist
        :param path_download(str): The full path of the location where you want to create the folder.
        :return:
        """
        # Checks if the directory does not exist
        if not os.path.exists(path_download):
            try:
                # Create the directory
                os.makedirs(path_download)
                print(f"Path {path_download} successfully created!")
            except OSError as e:
                # In case of error
                print(f"Error creating folder {path_download}: {e}")
        else:
            print(f"path {path_download} already exists")
    @staticmethod
    def function_list_path_files(path: str):
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
        """
        Once the SIM DOFET folder is selected, select the files from the desired years and save them in the indicated
        folder, with the name SIM_DOFET
        :param path_download: Path where the download should be made
        :param year: Years of interest
        :return:
        A folder with the name of the database and the years of interest will be saved in the specified path.
        """
        url = 'https://bigdata-arquivos.icict.fiocruz.br/PUBLICO/SIM_DOFET/ETLSIM.DOFET.zip'
        aa = self.function_download_data(url=url, path_download=path_download)
        if aa:
            # Identify the files in the folder
            path_ = f'{path_download}/ETLSIM.DOFET/dados/apache-airflow/data/SIM_DOFET'
            lista_arq = self.function_list_path_files(path=path_)
            # Filter addresses according to the years of interest
            enderecos_filtrados = [endereco for endereco in lista_arq if any(str(ano) in endereco for ano in year)]
            # Creating the folder in the path
            nova_pasta = f'{path_download}/SIM_DOFET'
            self.function_create_path(path_download=nova_pasta)
            # Copying the files to the new folder
            for i in enderecos_filtrados:
                shutil.copy(i, nova_pasta)
            # Delete the folder that was downloaded
            shutil.rmtree(f'{path_download}/ETLSIM.DOFET')
            print('SIM_DOFET folder created successfully')
        else:
            print(f'Check the url {url} or the way {path_download}')
        pass

    def function_import_sinasc(self, path_download: str, year: list):
        """
       Once the SINASC folder is selected, select the files from the desired years and save them in the indicated
       folder, with the name SINASC
       :param path_download: Path where the download should be made
       :param year: Years of interest
       :return:
       A folder with the name of the database and the years of interest will be saved in the specified path.
       """
        url = 'https://bigdata-arquivos.icict.fiocruz.br/PUBLICO/SINASC/ETLSINASC.zip'
        aa = self.function_download_data(url=url, path_download=path_download)
        if aa:
            # Identify the files in the folder
            path_ = f'{path_download}/ETLSINASC'
            lista_arq = self.function_list_path_files(path=path_)
            # Filter addresses according to the years of interest
            enderecos_filtrados = [endereco for endereco in lista_arq if any(str(ano) in endereco for ano in year)]
            # Creating the folder in the path
            nova_pasta = f'{path_download}/SINASC'
            self.function_create_path(path_download=nova_pasta)
            # Copying the files to the new folder
            for i in enderecos_filtrados:
                shutil.copy(i, nova_pasta)
            # Delete the folder that was downloaded
            shutil.rmtree(f'{path_download}/ETLSINASC')
            print('SINASC folder created successfully')
        else:
            print(f'Check the url {url} or the way {path_download}')
        pass

    def function_import_cnes(self, path_download: str, year: list):
        """
        Once the CNES folder is selected, select the files from the desired years and save them in the indicated
        folder, with the name CNES
        :param path_download: Path where the download should be made
        :param year: Years of interest
        :return:
        A folder with the name of the database and the years of interest will be saved in the specified path.
        """
        url = 'https://bigdata-arquivos.icict.fiocruz.br/PUBLICO/CNES/ETLCNES.zip'
        aa = self.function_download_data(url=url, path_download=path_download)
        if aa:
            # Identify the files in the folder
            path_ = f'{path_download}/ETLCNES'
            lista_arq = self.function_list_path_files(path=path_)
            year = [f'__{str(i)[2:]}_' for i in year]
            # Filter addresses according to the years of interest
            enderecos_filtrados = [endereco for endereco in lista_arq if any(str(ano) in endereco for ano in year)]
            # Creating the folder in the path
            nova_pasta = f'{path_download}/CNES'
            self.function_create_path(path_download=nova_pasta)
            # Copying the files to the new folder
            for i in enderecos_filtrados:
                shutil.copy(i, nova_pasta)
            # Delete the folder that was downloaded
            shutil.rmtree(f'{path_download}/ETLCNES')
            print('CNES folder created successfully')
        else:
            print(f'Check the url {url} or the way {path_download}')
        pass