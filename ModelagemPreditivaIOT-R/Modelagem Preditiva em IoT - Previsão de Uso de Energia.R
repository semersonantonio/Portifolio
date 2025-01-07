# Este projeto utiliza dados de uma rede de sensores IoT para criar modelos preditivos
# de consumo de energia de eletrodomésticos. Os dados foram coletados durante um
# período de cinco meses e incluem medições de temperatura, umidade e consumo de
# energia, além de informações meteorológicas da estação de um aeroporto próximo.
# O conjunto de dados foi processado, incluindo a filtragem de variáveis não preditivas
# e o tratamento de valores ausentes.

# O processo de análise e modelagem envolveu várias etapas. Inicialmente, foi realizada
# a análise exploratória dos dados, incluindo a visualização da distribuição do consumo
# de energia e a análise de correlação entre as variáveis. A filtragem das variáveis
# irrelevantes foi feita, excluindo colunas como data e variáveis aleatórias. Em seguida,
# os valores ausentes foram substituídos pela mediana (para variáveis numéricas) ou moda
# (para variáveis categóricas).

# Para a construção do modelo preditivo, foram treinados dois algoritmos de aprendizado
# de máquina: SVM (Máquinas de Vetores de Suporte) e XGBoost. Ambos os modelos foram ajustados
# utilizando validação cruzada para otimizar seus parâmetros. A importância das variáveis foi
# analisada com Random Forest, e as 10 variáveis mais relevantes foram selecionadas para a
# modelagem final.

# As métricas de desempenho (MAE, RMSE, R², MAPE, MedAE e RSE) foram calculadas para ambos os modelos,
# e o modelo com o menor erro absoluto médio (MAE) foi selecionado como o melhor. No final, as previsões
# do modelo escolhido foram aplicadas em um conjunto de teste, e os resultados foram exportados para um
# arquivo CSV.

# Este processo demonstrou a eficácia da combinação de técnicas de aprendizado de máquina e IoT
# na previsão de consumo de energia, com a possibilidade de otimizar o uso de energia com base
# em variáveis ambientais e de operação.
# ____________________________________________________________________

# Imports
library(dplyr)
library(ggplot2)
library(corrplot)
library(caret)
library(randomForest)
library(e1071)
library(xgboost)
library(DataExplorer)
library(reshape2)

# Carregando os dados e análise exploratória
dados <- read.csv("data/training.csv")
str(dados)
summary(dados)

# Convertendo a coluna de data
dados$date <- as.POSIXct(dados$date, format = "%Y-%m-%d %H:%M:%S")

# Plot para exploração dos dados
plot_intro(dados)
plot_missing(dados)

# Distribuição do consumo de energia
ggplot(dados, aes(x = Appliances)) +
  geom_histogram(binwidth = 50, fill = "blue", color = "black") +
  labs(title = "Distribuição do Consumo de Energia", x = "Consumo de Energia", y = "Frequência")

# Verificando as correlações das variáveis numéricas
correlacao <- cor(dados %>% select(-date, -WeekStatus, -Day_of_week))
View(correlacao)

# Em plot
corrplot(correlacao, method = "color", type = "upper", tl.cex = 0.7)


# Pré-Processamento dos Dados

# Filtrando as variáveis irrelevantes ou não preditivas
dados_filtrados <- dados %>%
  select(-date, -WeekStatus, -Day_of_week, -rv1, -rv2)

# Tratando valores ausentes e substituindo NA pela mediana
dados_filtrados <- dados_filtrados %>% mutate(across(everything(), ~ {
  if (is.numeric(.)) {
    # Mediana para colunas numéricas
    ifelse(is.na(.), median(., na.rm = TRUE), .)
  } else {
    # Moda para colunas categóricas
    ifelse(is.na(.), unique(.)[which.max(tabulate(match(., unique(.))))], .)
  }
}))

# Verificando se não há mais valores ausentes
any(is.na(dados_filtrados))

# Separando X e y
X <- dados_filtrados %>% select(-Appliances)
y <- dados_filtrados$Appliances

# Divisão em treino e validação
set.seed(42)
indice_treino <- createDataPartition(y, p = 0.8, list = FALSE)
dados_treino <- dados_filtrados[indice_treino, ]
dados_validacao <- dados_filtrados[-indice_treino, ]

# Selecionando os melhores atributos com Random Forest
modelo_rf <- randomForest(Appliances ~ ., data = dados_treino, importance = TRUE)
importancia <- as.data.frame(varImp(modelo_rf))

# Selecionando as 10 variáveis mais importantes
features_selecionadas <- rownames(importancia)[order(-importancia$Overall)][1:10]
dados_treino <- dados_treino[, c("Appliances", features_selecionadas)]
dados_validacao <- dados_validacao[, c("Appliances", features_selecionadas)]

# Visualizando importância das variáveis
importancia$Variable <- rownames(importancia)
ggplot(importancia, aes(x = reorder(Variable, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "blue") +
  coord_flip() +
  labs(title = "Importância das Variáveis (Random Forest)", x = "Variáveis", y = "Importância")


# Treinando os modelos com validação cruzada
# Configuração para validação cruzada
controle <- trainControl(method = "cv", number = 5)

# Modelo 1 - SVM com otimização
tune_grid_svm <- expand.grid(C = c(0.1, 1, 10), sigma = c(0.01, 0.1, 1))
set.seed(42)
modelo_svm <- train(Appliances ~ ., data = dados_treino, method = "svmRadial",
                    tuneGrid = tune_grid_svm, trControl = controle)

# Modelo 2 - XGBoost com otimização
# Usando o xgb.DMatrix diretamente
train_matrix_data <- as.matrix(dados_treino[, -1])
train_labels <- dados_treino$Appliances

# Criando o xgb.DMatrix
train_matrix <- xgb.DMatrix(data = train_matrix_data, label = train_labels)

# Definindo parâmetros
param_xgb <- list(
  eta = 0.1,                       
  max_depth = 6,                   
  nrounds = 100,                   
  objective = "reg:squarederror",  
  eval_metric = "rmse"             
)

# Treinando o modelo XGBoost
modelo_xgb <- xgboost(
  data = train_matrix, 
  params = param_xgb, 
  nrounds = 400
)


# Avaliando os modelos no conjunto de validação

# Predições no conjunto de validação
# SVM
predicoes_svm <- predict(modelo_svm, dados_validacao)

# XGBoost
validacao_matrix <- xgb.DMatrix(data = as.matrix(dados_validacao[, -1]))
predicoes_xgb <- predict(modelo_xgb, validacao_matrix)

# Função para calc. as métricas
calcular_metricas <- function(y_real, y_pred) {
  # Erro absoluto médio
  mae <- mean(abs(y_pred - y_real))
  
  # Raiz do erro quadrático médio
  rmse <- sqrt(mean((y_pred - y_real)^2))
  
  # R² (coeficiente de determinação)
  sst <- sum((y_real - mean(y_real))^2)
  sse <- sum((y_real - y_pred)^2)
  r2 <- 1 - (sse / sst)
  
  # Erro percentual médio absoluto
  mape <- mean(abs((y_real - y_pred) / y_real)) * 100
  
  # Mediana do erro absoluto
  medae <- median(abs(y_pred - y_real))
  
  # Erro padrão residual
  n <- length(y_real)
  rse <- sqrt(sse / (n - 1))
  
  # Retornar todas as métricas em um data.frame
  data.frame(MAE = mae, RMSE = rmse, R2 = r2, MAPE = mape, MedAE = medae, RSE = rse)
}

# Métricas do modelo SVM
metricas_svm <- calcular_metricas(dados_validacao$Appliances, predicoes_svm)

# Métricas do modelo XGBoost
metricas_xgb <- calcular_metricas(dados_validacao$Appliances, predicoes_xgb)

# Comparação das métricas
metricas_comparacao <- rbind(SVM = metricas_svm, XGBoost = metricas_xgb)
print(metricas_comparacao)

# Escolhendo o melhor modelo com base no MAE
melhor_modelo <- rownames(metricas_comparacao)[which.min(metricas_comparacao$MAE)]
cat("Melhor modelo com base no MAE:", melhor_modelo, "\n")

# Se o melhor modelo for o SVM, se não XGBoost
if (melhor_modelo == "SVM") {
  modelo_final <- modelo_svm
} else {
  modelo_final <- modelo_xgb
}

# Previsões do modelo final no conjunto de teste
dados_teste <- read.csv("data/testing.csv")
dados_teste <- dados_teste[, c("Appliances", features_selecionadas)]
teste_matrix <- xgb.DMatrix(data = as.matrix(dados_teste[, -1]))

# Fazendo as previsões com o modelo final
if (melhor_modelo == "SVM") {
  predicoes_final <- predict(modelo_final, dados_teste)
} else {
  predicoes_final <- predict(modelo_final, teste_matrix)
}

# Cria um data frame com os resultados reais e previstos
resultado_final <- data.frame(
  Appliances_Real = dados_teste$Appliances,
  Appliances_Previsto = predicoes_final
)
View(resultado_final)

# Salvando os resultados em um arquivo csv
write.csv(resultado_final, "previsoes_final.csv", row.names = FALSE)
