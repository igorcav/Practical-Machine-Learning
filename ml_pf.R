### Bibliotecas (Libraries)
library(caret)
library(dplyr)
library(knitr)
library(corrplot)
library(RColorBrewer)

### Leitura dos dados (Reading data)
setwd("/home/quarteto/Downloads")
treino <- read.csv("pml-training.csv", header = T, stringsAsFactors = F) 
teste  <- read.csv("pml-testing.csv", header = T, stringsAsFactors = F)
dim(treino)
dim(teste)
head(treino)
str(treino)

### Limpeza dos dados (Cleaning data)
treino <- treino[, -nearZeroVar(treino)]
teste  <- teste[, -nearZeroVar(teste)]
treino <- treino[, (colSums(is.na(treino)) == 0)]
teste  <- teste[, (colSums(is.na(teste)) == 0)]
treino <- select(treino, -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, num_window))
teste  <- select(teste, -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, num_window))
treino$classe <- as.factor(treino$classe)

### Correlação (Correlation)
cor(treino[,-ncol(treino)]) %>%
  corrplot(type="upper", order="hclust", col=brewer.pal(n=8, name="RdYlBu"))

### Divisao (partioning)
set.seed(1111) 
emTreino   <- createDataPartition(treino$classe, p = 0.70, list = FALSE)
validacao  <- treino[-emTreino, ]
treino     <- treino[emTreino, ]

### Predição com método GBM (Prediction with GBM method)
mod_gbm  <- train(classe ~., data = treino, method = "gbm")
pred_gbm <- predict(mod_gbm, validacao)
confMatrix <- confusionMatrix(validacao$classe, pred_gbm)

### Acurácia do modelo (Model accuracy)
confMatrix$table %>%
  plot(main = paste("Method GBM\nAccuracy:", confMatrix$overall['Accuracy']), col="orange")

### Predição do modelo (Model prediction)
predict(mod_gbm, teste)
# [1] B A B A A E D B A A B C B A E E A B B B
