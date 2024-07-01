# install.packages("read.dbc")

library(read.dbc)

# Caminho para o arquivo .dbc
file_path <- 'C:\\Users\\gabri\\Documents\\PSW_COVID_OBITO_FETAL\\SIM_DOFET\\DOFET22.DBC'

# Tenta ler o arquivo .dbc
df <- read.dbc(file_path)

# Verifica as primeiras linhas do DataFrame
head(df)

length(df)

write.csv2(df, 'C:\\Users\\gabri\\Documents\\PSW_COVID_OBITO_FETAL\\SIM_DOFET\\SIM_DOFET_2022_sujo.csv')
