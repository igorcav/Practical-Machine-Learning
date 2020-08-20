# Practical Machine Learning - Coursera Final Project

*Igor Dutra Cavalcante, 19 de agosto de 2020*

### Dados iniciais (Initial data)
The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

### Bibliotecas (Libraries)
```{r}
library(caret)
library(dplyr)
library(corrplot)
library(RColorBrewer)
```

First, we must download and clean the data.
### Leitura dos dados (Reading data)
```{r}
treino <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", header = T, stringsAsFactors = F) 
teste  <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", header = T, stringsAsFactors = F)
dim(treino)
dim(teste)
head(treino)
str(treino)
```

### Limpeza dos dados (Cleaning data)
```{r}
treino <- treino[, -nearZeroVar(treino)]
teste  <- teste[, -nearZeroVar(teste)]
treino <- treino[, (colSums(is.na(treino)) == 0)]
teste  <- teste[, (colSums(is.na(teste)) == 0)]
treino <- select(treino, -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, num_window))
teste  <- select(teste, -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, num_window))
treino$classe <- as.factor(treino$classe)
```

Once the data is clean and the features with no predictive value are discarded, we will check the correlation strength between the remaining features.
### Correlação (Correlation)
```{r}
cor(treino[,-ncol(treino)]) %>%
  corrplot(type="upper", order="hclust", col=brewer.pal(n=8, name="RdYlBu"))
```

![] https://github.com/igorcav/Practical-Machine-Learning/blob/master/Correlacao.jpeg
  
The cross-validation method was then used. This method separates the training data (70%) and validation (30%), allowing for better model validation.
### Divisão (Partioning)
```{r}
set.seed(1111) 
emTreino   <- createDataPartition(treino$classe, p = 0.70, list = FALSE)
validacao  <- treino[-emTreino, ]
treino     <- treino[emTreino, ]
```

The model was then trained using the Generalized Boosted Model method.
### Predição com método GBM (Prediction with GBM method)
```{r}
mod_gbm  <- train(classe ~., data = treino, method = "gbm")
pred_gbm <- predict(mod_gbm, validacao)
confMatrix <- confusionMatrix(validacao$classe, pred_gbm)
```

[id]: https://github.com/igorcav/Practical-Machine-Learning/Acuracia.jpeg  "Accuracy"

Once the model has been trained, the validation shows an accuracy of approximately 96%. This implies an error of approximately 3.6%. That is, the model, when trying to predict classes, has this chance to make a mistake. It should be noted, as shown in the graph below, that this error is evenly distributed across classes.
```{r}
confMatrix$table %>%
  plot(main = paste("Method GBM\nAccuracy:", confMatrix$overall['Accuracy']), col="orange")
```

Finally, let's test the model in the test cases:
```{r}
predict(mod_gbm, teste)
# [1] B A B A A E D B A A B C B A E E A B B B
```
