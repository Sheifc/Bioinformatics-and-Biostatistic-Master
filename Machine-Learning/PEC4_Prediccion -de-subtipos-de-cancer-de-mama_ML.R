---
title: "Predicción de subtipos de cáncer de mama"
subtitle: "PEC4 Machine Learning" 
author: "Sheila Fernández Cisneros"
date: '`r format(Sys.Date(),"%e de %B, %Y")`'
output: 
  html_document: 
    toc: TRUE
    toc_depth: 2
    toc_float: TRUE
    theme: cosmo
    number_section: TRUE
  pdf_document: 
    toc: TRUE
    toc_depth: 2
    number_section: TRUE
---

```{r setup, include=FALSE}
library(knitr)

options(max.print="75")
opts_chunk$set(echo=FALSE,
	             cache=TRUE,
               prompt=FALSE,
               tidy=TRUE,
               comment=NULL,
               message=FALSE,
               warning=FALSE)
opts_knit$set(width=75) 
```


```{r libraries, include=FALSE}
# Cargamos las librerías 
library(knitr)

library(knitr)
library(stringr)
library(class)
library(dplyr)
library(gmodels)
library(ggseqlogo)
library(ggplot2)
library(ROCR)
library(pROC)

# Importamos los paquetes necesarios para llamar el paquete caret:
library(ggplot2)
library(lattice)
library(caret)
# Naive Bayes
library(e1071)
# SVM
library(kernlab)
# Decision Tree
library(C50)
# Random Forest
library(randomForest)
```

<style>
body {text-align: justify}
</style>

# Introducción 

El cáncer de mama humano es una enfermedad heterogénea en términos de alteraciones moleculares, composición celular y cambios clínicos. Los tumores de mama se pueden clasificar en varios subtipos, según los niveles de expresión de mRNA. En esta actividad consideramos un subconjunto de datos ómicos generados por The Cancer Genome Atlas Network (TCGA). Los datos corresponden a 150 muestras de RNAseq, que han sido normalizados y prefiltrados. El objetivo de este análisis es identificar un modelo que discrimine los subtipos de cáncer de mama Basal, Her2 y LumA. Los datos se encuentran en el fichero rna.csv. La primera columna contiene el subtipo de cáncer y el resto de columnas la expresión normalizada de genes preseleccionados.

# Objetivo

En esta PEC se analizan estos datos mediante la implementación de los diferentes algoritmos estudiados: k-Nearest Neighbour, Naive Bayes, Artificial Neural Network, Support Vector Machine, Trees y Random Forest para predecir el subtipo de cáncer.

Para ello, se realizará en R con el entorno R studio la implementación de los algoritmos k-Nearest Neighbour, Naive Bayes, Support Vector Machine, Trees y Random Forest incluídos en este documento y en Python a través de google colab, el algoritmo Artificial Neural Network que se aportará en otro documento adicional.

# Puntos importantes:

* Realizar una exploración de los datos que incluya una estadística descriptiva básica de las variables mediante tablas y gráficos (heatmap,...).

* En cada algoritmo hay que realizar las siguientes tres etapas: 1) Transformación de los datos (en caso necesario) 2) Entrenar el modelo 3) Predicción y Evaluación del algoritmo. En la fase 2) "tunear" diferentes valores de los hiperparámetros del algoritmo para posteriormente evaluar su rendimiento.

* Se debe aplicar la misma selección de datos training y test en todos los algoritmos. Utilizando la misma semilla aleatoria, para separar los datos en dos partes, una parte para training (75%) y otra parte para test (25%).

* En todos los casos se evalua la calidad del algoritmo con la información obtenida de la función confusionMatrix() del paquete caret, o equivalente en Python.

* Para la ejecución específica de cada algoritmo en R se puede usar la función de cada algoritmo como se presenta en el libro de referencia o usar el paquete caret con los diferentes modelos de los algoritmos. O incluso, hacer una versión mixta.

* Comentario sobre el informe dinámico en Rmd. Una opción interesante del knitr es poner cache=TRUE. Por ejemplo: knitr::opts_chunk$set(echo = FALSE, comment = NULL, cache = TRUE)
Con esta opción al ejecutar el informe dinámico crea unas carpetas donde se guardan los resultados de los procesos. Cuando se vuelve a ejecutar de nuevo el informe dinámico solo ejecuta código R donde se ha producido cambios, en el resto lee la información previamente descargada. Es una opción muy adecuada cuando la ejecución es muy costosa computacionalmente.

# Importación de los datos:

Comenzamos importando los datos `rna.csv` a un objeto `data` con la función read.csv() separado por comas.

```{r}
# Importamos los datos usando read.csv()
data <- read.csv("rna.csv", sep=",")
```

# Análisis exploratorio y descriptiva estadística

Confirmamos que el dataset `data` está estructurado en `r nrow(data)` muestras de RNAseq y `r ncol(data)` variables de expresión. 

```{r}
# Mostramos las dimensiones de nuestro dataset data con dim()
dim(data)
print(paste("De las cuales se presentan",dim(data)[2] - 1, "variables predictoras y 1 variable respuesta Y."))
```
Para inspeccionar la estructura de nuestras variables del dataset usando la función str() y evitar el truncamiento usaremos el argumento `list.len` con el cual mostraremos la salida de la lista completa cuando se usa la función str en R, es decir, sin truncar la parte inferior de la lista.

```{r}
str(data, list.len = ncol(data))
```

Observamos que todas las variables son numéricas a excepción de la variable `Y` que es categórica. 

La variable respuesta `Y` o tipo de cáncer de mama presenta 3 niveles: Basal, Her2 y LumA. La convertimos a factor y mostramos dicha conversión y una tabla con el recuento de las variables en función del tipo de cáncer de mama (Basal, Her2 y LumA). 

```{r}
data$Y <- factor(data$Y)
levels(data$Y) <- c("Basal", "Her2", "LumA")
str(data$Y)
table(data$Y)
```


Continuaremos mostrando una descriptiva estadística. Debido al alto volumen de variables incluído en `data`, mostramos a modo de ejemplo las 10 primeras variables. 

```{r}
# Obtenemos un resumen estadístico con la función summary()
summary(data[, 1:10])
```

Mostramos el mínimo de los valores de los datos para identificar si hay variables sin expresión.

```{r}
min(data[-1])
```

* **Representación gráfica:** debido al alto volumen de variables, vamos a visualizar las 10 primeras mediante un gráfico de cajas. 

Como ya sabemos, si la mediana se sitúa en el centro de la caja entonces la distribución es simétrica y tanto la media, mediana y moda coinciden.

Si la mediana corta la caja en dos lados desiguales se tiene:

-Asimetría positiva o segada a la derecha si la parte más larga de la caja es la parte superior a la mediana. Los datos se concentran en la parte inferior de la distribución. La media suele ser mayor que la mediana.

-Asimetría negativa o sesgada a la izquierda si la parte más larga es la inferior a la mediana. Los datos se concentran en la parte superior de la distribución. La media suele ser menor que la mediana.

Entre las variables representadas, tenemos la variable respuesta (tipo de cáncer de mama) que presenta asimetría negativa. Las variables predictoras representadas presentan una distribución más o menos centrada con algunas ligeras variaciones y algunos valores extremos. 

```{r}
boxplot(data[1:10], main = "Boxplot de las 10 primeras variables", col="lightblue")
```
Comprobamos que la representación de un heatmap en este caso para tantas variables no es muy ilustrativo. 

```{r}
heatmap(as.matrix(data[, -1]))
```

# Separación de datos en los conjuntos train y test

Realizamos la separación entre conjunto **train** y **test**, que nos servirán para entrenar y evaluar los modelos, respectivamente. Lo haremos fijando la semilla `set.seed` para que la separación de los datos sea la misma cada vez que ejecutamos el código. Este código ha sido desarrollado de forma dinámica, de manera que está adaptado a cualquier conjunto de datos.

```{r}
#Fijamos semilla:
set.seed(12345)

# Mezclamos el conjunto de datos para evitar sesgos en las separación train/test:
data <- data[order(runif(nrow(data))), ]

#Determinamos el valor límite para que el informe sea dinámico. Este valor será la cantidad de muestras que formarán el conjunto train (un 75% de los datos).
limit <- round(nrow(data)*0.75)
limit

#Separamos los datos en el conjunto de validación (test) y el conjunto de entrenamiento (train):
train <- data[1:limit,]
test <- data[(limit+1):(nrow(data)),]
```

Podemos asegurarnos que la separación ha sido homogénea representando la distribución de la variable respuesta `Y`. Como vemos, en ambos conjuntos de datos hay aproximadamente la misma proporción de genes para cada subtipo de cáncer.

```{r}
# Distribución de la variable respuesta: 
table(train$Y)/nrow(train)
table(test$Y)/nrow(test)
```

# Aplicación de algoritmos para la clasificación

Para la ejecución específica de cada algoritmo se puede usar la función de cada algoritmo como se presenta en el libro de referencia o usar el paquete caret con los diferentes modelos de los algoritmos. 

En este caso se realizará el análisis con los algoritmos de manera mixta: primero el paquete caret para tener una visión gobal de las métricas para cada opción, y se comprobará dichas métricas manualmente con las funciones de cada paquete. 

# K-Nearest Neighbour

Se explorarán los valores para el número de vecinos k = 1, 3, 5, 7, 11.

## Step 1: Transformación de los datos. 

En el caso del algoritmo k-NN, se usan los datos normalizados en las variables cuantitativas y con variables dummy en las categóricas. En este caso ya los tenemos normalizados. No hay que hacer transformación de los datos. 

## Step 2 y Step 3: Entrenar el modelo, Predicción y Evaluación del algoritmo. 

```{r}
# Fijamos semilla
set.seed(123)

# Indicamos los valores de k que queremos evaluar:
knn_grid <- expand.grid(.k = c(1, 3, 5, 7, 11))

# Indicamos la separación entre conjunto train y test (lo utilizaremos de la misma manera en el resto de algoritmos):
ctrl <- trainControl(method = "LGOCV", p = 0.75)

knn <- train(Y ~ ., 
             data = data, 
             method = "knn", 
             trControl = ctrl,
             tuneGrid = knn_grid)
knn

# Evaluación del modelo con predict():
knn_caret_pred <- predict(knn, data)
table(knn_caret_pred, data$Y)
```

En este caso el valor de k que mejor clasificación realiza según métricas como la precisión y el valor de Kappa es **k = 3**. Veamoslo con más detalle: 

Aplicamos el algoritmo k-NN para cada valor de k y mostramos las métricas para evaluar cada predicción con las `confusionMatrix` del paquete caret.

```{r}
test1 <- test[-1] # eliminamos la columna Y
train1 <- train[-1] # eliminamos la columna Y
```

* **K=1**

```{r}
set.seed(12345) # fijamos una semilla
pred1 <-knn(train = train1, test = test1, cl = train$Y, k = 1, prob = TRUE)
pred1
cm1 <- confusionMatrix(pred1, test$Y, positive = "1")
cm1
```

```{r}
set.seed(12345) # fijamos una semilla
pred3 <-knn(train = train1, test = test1, cl = train$Y, k = 3, prob = TRUE)
pred3
cm3 <- confusionMatrix(pred3, test$Y, positive = "1")
cm3
```

```{r}
set.seed(12345) # fijamos una semilla
pred5 <-knn(train = train1, test = test1, cl = train$Y, k = 5, prob = TRUE)
pred5
cm5 <- confusionMatrix(pred5, test$Y, positive = "1")
cm5
```

```{r}
set.seed(12345) # fijamos una semilla
pred7 <-knn(train = train1, test = test1, cl = train$Y, k = 7, prob = TRUE)
pred7
cm7 <- confusionMatrix(pred7, test$Y, positive = "1")
cm7
```

```{r}
set.seed(12345) # fijamos una semilla
pred11 <-knn(train = train1, test = test1, cl = train$Y, k = 11, prob = TRUE)
pred11
cm11 <- confusionMatrix(pred11, test$Y, positive = "1")
cm11
```

## Resultados

```{r}
# Resultados k-NN con caret -> seleccionamos las medidas kappa y accuracy. 
knn_caret_results <- (knn$results)[c(2,3)]

# Cambiamos los nombres de las filas y las columnas:
rownames(knn_caret_results) <- c("knn=1", "knn=3", "knn=5", "knn=7", "knn=11")
colnames(knn_caret_results) <- c("Accuracy_caret", "Kappa_caret")

# Resultados con el algoritmo k-NN:

k1 <- data.frame(Accuracy_knn = (cm1$overall)[1],
                 Kappa_knn = (cm1$overall)[2],
                 Sensitivity = (cm1$byClass)[1],
                 Specificity = (cm1$byClass)[2])

k3 <- data.frame(Accuracy_knn = (cm3$overall)[1],
                 Kappa_knn = (cm3$overall)[2],
                 Sensitivity = (cm3$byClass)[1],
                 Specificity = (cm3$byClass)[2])

k5 <- data.frame(Accuracy_knn = (cm5$overall)[1],
                 Kappa_knn = (cm5$overall)[2],
                 Sensitivity = (cm5$byClass)[1],
                 Specificity = (cm5$byClass)[2])

k7 <- data.frame(Accuracy_knn = (cm7$overall)[1],
                 Kappa_knn = (cm7$overall)[2],
                 Sensitivity = (cm7$byClass)[1],
                 Specificity = (cm7$byClass)[2])

k11 <- data.frame(Accuracy_knn = (cm11$overall)[1],
                 Kappa_knn = (cm11$overall)[2],
                 Sensitivity = (cm11$byClass)[1],
                 Specificity = (cm11$byClass)[2])


# Unimos resultados y cambiamos el nombre de las filas:
knn_results <- rbind(k1, k3, k5, k7, k11)

rownames(knn_results) <- c("knn_k1", "knn_k3", "knn_k5", "knn_k7", "knn_k11")

# Unimos los resultados del paquete caret y los del algoritmo knn:
knn_metrics <- cbind(knn_caret_results, knn_results)
knn_metrics
```

# Naive Bayes

En el caso del algoritmo de naive Bayes no es necesario normalizar los datos. Esto es debido a que las características del conjunto de datos deben ser categóricas cuando queremos aplicar este método. Cuando son numéricas símplemente son categorizados en varios grupos, en un proceso llamado *"binning"*.

En este apartado se explorará la aplicación del algoritmo Naive Bayes con la opción de activar o no laplace. 

## Step 1: Transformación de los datos.

No es necesario transformar los datos nuevamente. 

## Step 2 y Step 3: Entrenar el modelo, Predicción y Evaluación del algoritmo.

* **Función naiveBayes del paquete `e1071`**

Generamos el modelo Naive Bayes a partir de la función `naiveBayes` del paquete `e1071` con laplace 0 y 1.

**laplace = 0:**

```{r}
set.seed(12345)
# Con la función naivebayes() creamos el modelo:
mod_nb1 <- naiveBayes(Y~., train, laplace=0)
pred_nb1 <- predict(mod_nb1, test)
```

**laplace = 1:**

```{r}
set.seed(12345)
mod_nb2 <- naiveBayes(Y~., train, laplace=1)
pred_nb2 <- predict(mod_nb2, test)
```

**Evaluación de la predicción**

Para ello utilizaremos la `confussionMatrix()` del paquete `caret`:

- Evaluación de la predicción con `laplace = 0`:
```{r}
nb_results_lapl0 <- confusionMatrix(pred_nb1, test$Y, positive = "1")
```

- Evaluación de la predicción con `laplace = 1`:
```{r}
nb_results_lapl1 <- confusionMatrix(pred_nb2, test$Y, positive = "1")
```

## Resultados

Como se puede observar, los valores de las métricas entre el modelo con o sin laplace son iguales. De manera que no existen diferencias en el caso de este conjunto de datos para diagnosticar el tipo de cáncer de mama en el caso de usar o no laplace. 

```{r}
# Montamos los resultados del algoritmo naive bayes:
nb_r_laplace0 <- data.frame(Accuracy = (nb_results_lapl0$overall)[1],
                 Kappa = (nb_results_lapl0$overall)[2],
                 Sensitivity = (nb_results_lapl0$byClass)[1],
                 Specificity = (nb_results_lapl0$byClass)[2])

nb_r_laplace1 <- data.frame(Accuracy = (nb_results_lapl1$overall)[1],
                 Kappa = (nb_results_lapl1$overall)[2],
                 Sensitivity = (nb_results_lapl1$byClass)[1],
                 Specificity = (nb_results_lapl1$byClass)[2])

nb_results <- rbind(nb_r_laplace0,
                    nb_r_laplace1)

rownames(nb_results) <- c("Naive_bayes_laplace_0", "Naive_bayes_laplace_1")

nb_results
```

# Artifitial Neural Network

Se explorarán las arquitecturas densas con dos capas ocultas de: 
1) 20 y 10 nodos, 2) 50 y 10 nodos.

Se podría hacer el ANN con el modelo mlpML de caret, que es multicapa pero en este caso vamos a realizar este algoritmo en `google colab` con Python. Archivo que se adjunta conjuntamente. Posteriormente los resultados serán comentados en su totalidad en la conclusión y discusión de los resultados en este informe. 

# Support Vector Machine

## Step 1: Transformación de los datos: 

No es necesario transformar los datos nuevamente.

## Step 2: Entrenar el modelo.

Entrenaremos el modelo y realizaremos la predicción. Lo haremos utilizando dos núcleos o kernels distintos: **lineal y Gaussiano.**

* **Kernel lineal** -> opción `vanilladot`del parámetro *kernel*. 

```{r}
# Entrenamos el modelo con la opción vanilladot:
linear_classifier <- ksvm(Y ~ ., 
                   data = train, 
                   kernel = "vanilladot")
linear_classifier
```

* **Kernel Gaussiano** -> opción `rbfdot` del parámetro *kernel*. 

```{r}
gaussian_classifier <- ksvm(Y ~ ., 
                   data = train, 
                   kernel = "rbfdot")
gaussian_classifier
```

## Step 3: Predicción y Evaluación del algoritmo.

**Evaluación de la predicción**

Utilizamos la función `predict` para usar el modelo clasificador y hacer predicciones en el grupo test. Creamos una predicción para cada tipo de clasificador (linear o Gaussiano):

```{r}
linear_pred <- predict(linear_classifier, test)
gaussian_pred <- predict(gaussian_classifier, test)
```

Para evaluar el modelo utilizaremos una matriz de confusión: 

* **Predicción linear:**
```{r}
svm_linear_cm <- confusionMatrix(linear_pred, test$Y, positive = "1")
```
* **Predicción guassiana:**
```{r}
svm_gaus_cm <- confusionMatrix(gaussian_pred, test$Y, positive = "1")
```

## Resultados
 
```{r}
svm_linear <- data.frame(Accuracy = (svm_linear_cm$overall)[1],
                 Kappa = (svm_linear_cm$overall)[2],
                 Sensitivity = (svm_linear_cm$byClass)[1],
                 Specificity = (svm_linear_cm$byClass)[2])

svm_gauss <- data.frame(Accuracy = (svm_gaus_cm$overall)[1],
                 Kappa = (svm_gaus_cm$overall)[2],
                 Sensitivity = (svm_gaus_cm$byClass)[1],
                 Specificity = (svm_gaus_cm$byClass)[2])

svm_results <- rbind(svm_linear,
                    svm_gauss)

rownames(svm_results) <- c("SVM_linear_kernel", "SVM_gaussian_kernel")

svm_results
```
# Decision Tree

## Step 1: Transformación de los datos: 

No es necesario transformar los datos.  

## Step 2 y Step 3: Entrenar el modelo, Predicción y Evaluación del algoritmo. 

Se explorará la opción de activar o no boosting.

* **Sin *boosting*:** Para ello, dejamos el parametro trials = 1 como predeterminado. 

```{r}
set.seed(12345)
# Entranamos el modelo:
tree <- C5.0(train1, train$Y)
tree

# Podemos ver las decisiones del arbol de decisión con summary():
summary(tree)
```

Este resumen del modelo también nos muestra una matriz de confusión, indicando el error del modelo, en este caso un 2.7% de error. Evaluaremos el modelo realizando una matriz de confusión que nos muestre todas las métricas: 

```{r}
# Predicción: 
tree_pred <- predict(tree, test)

# Matriz de confusión con sus métricas:
tree_cm <- confusionMatrix(tree_pred, test$Y, positive = "1")

tree_cm
```

* **Con *boosting*:** Esta vez indicaremos el parametro trials como 10 para activar la función de boosting. Como vemos en el resumen del modelo con boosting, para cada trial se prueban unos límites de los diferentes valores de variables predictoras hasta encontrar los que determinen un modelo óptimo. 

```{r}
set.seed(12345)
# Modelo:
tree_boost <- C5.0(train1, train$Y, trials=10)

# Vemos le resumen del modelo para ver las decisiones: 
summary(tree_boost)
```

**Evaluamos el modelo**

```{r}
# Predicción: 
tree_boost_pred <- predict(tree_boost, test)

# Matriz de confusión con sus métricas:
tree_boost_cm <- confusionMatrix(tree_boost_pred, test$Y, positive = "1")

tree_boost_cm
```

## Resultados 

Unimos los resultados y las métricas en un mismo dataframe:

```{r}
tree_res <- data.frame(Accuracy = (tree_cm$overall)[1],
                       Kappa = (tree_cm$overall)[2],
                       Sensitivity = (tree_cm$byClass)[1],
                       Specificity = (tree_cm$byClass)[2])

tree_boost_res <- data.frame(Accuracy = (tree_boost_cm$overall)[1],
                       Kappa = (tree_boost_cm$overall)[2],
                       Sensitivity = (tree_boost_cm$byClass)[1],
                       Specificity = (tree_boost_cm$byClass)[2])

tree_results <- rbind(tree_res, tree_boost_res)

rownames(tree_results) <- c("Decision_tree", "Decision_tree_boosting")

tree_results
```
# Random Forest

## Step 1: Transformación de los datos: 

No es necesario transformar los datos nuevamente.

## Step 2 y 3: Entrenar el modelo, Predicción y Evaluación del algoritmo. 

Se explorará la opción de número de árboles n = 50, 100.

* **Paquete caret:** Primero utilizaremos la función `train` para determinar que número de variable hace que el modelo sea óptimo, probando con 2, 4 y 8 variables. 

```{r}
set.seed(12345)
# En este caso probaremos variando el numero de variables que el random forest usara en cada caso:
rf_grid <- expand.grid(.mtry = c(2,4,8))

# Cargamos el modelo: 
rf_caret <- train(Y~., data = data, method = "rf", trControl = ctrl, tuneGrid = rf_grid)

rf_caret
```

El mayor valor de accuracy se da cuando se tienen en cuenta 8 variables. 

* **Paquete Random Forest:** Realizamos el análisis con la función del algoritmo utilizando el paquete `randomForest` con 50 y 100 árboles de decisión. 

**100 árboles:** Para ello fijamos el parámetro `ntree` a 50. 

```{r}
set.seed(12345)

# Cargamos la librería randomForest:


# Construimos el modelo:
rf_50 <- randomForest(Y ~., data = data, ntree = 50)
rf_50

# Evaluamos el modelo:

rf_50_pred <- predict(rf_50, test1)
rf_50_cf <- confusionMatrix(rf_50_pred, test$Y, positive = "1")

rf_50_cf
```
**200 árboles:** Para ello fijamos el parametro `ntree` a 100. 

```{r}
set.seed(12345)

# Construimos el modelo:
rf_100 <- randomForest(Y ~., data = data, ntree = 100)
rf_100

# Evaluamos el modelo:

rf_100_pred <- predict(rf_100, test1)
rf_100_cf <- confusionMatrix(rf_100_pred, test$Y, positive = "1")

rf_100_cf
```

## Resultados

```{r}
rf_50_res <- data.frame(Accuracy = (rf_50_cf$overall)[1],
                       Kappa = (rf_50_cf$overall)[2],
                       Sensitivity = (rf_50_cf$byClass)[1],
                       Specificity = (rf_50_cf$byClass)[2])

rf_100_res <- data.frame(Accuracy = (rf_100_cf$overall)[1],
                       Kappa = (rf_100_cf$overall)[2],
                       Sensitivity = (rf_100_cf$byClass)[1],
                       Specificity = (rf_100_cf$byClass)[2])

rf_results <- rbind(rf_50_res, rf_100_res)

rownames(rf_results) <- c("Random_forest_50", "Random_forest_100")

rf_results
```
# Conclusión y discusión

Sección de conclusión y discusión sobre el rendimiento de los algoritmos con al menos tres métricas para el problema tratado. Proponer qué modelo o modelos son los mejores. 

## Resultados

Comenzaremos uniendo todos los resultados:

```{r}
# Los resultados de caret no pueden inurse con el resto, ya que tienen el musmo número de columnas:
knn_caret_results

# Cambiamos el nombre de los resultados del algoritmo knn para poder unir todos los resultados:
colnames(knn_results) <- c("Accuracy", "Kappa", "Sensitivity", "Specificity")

# Unimos los resultados de los diferentes algoritmos:
all_results <- rbind(knn_results, nb_results, svm_results, tree_results, rf_results)
all_results
```

Observamos en la tabla las métricas para evaluar el rendimiento de la predicción en los diferentes algoritmos. 

* **Accuracy:**

Comenzaremos viendo la precisión o accuracy, ya que es una métrica que comparten todos los algoritmos.

Como vemos, los algoritmos que proporcionan el mayor accuracy son los **Random Forest**, tanto de 50 árboles como de 100. Este algoritmo consiguió predecir correctamente todos los valores del conjunto test de manera óptima. Le sigue, con un accuracy de **0.97**, el algoritmo de *Naive Bayes* y *SVM* en sus ambas versiones estudiadas, y a continuación tenemos el algoritmo *KNN*, con un accuracy de **0.92** en todos sus hipermarámetros. Como algoritmos con accuracy más bajo tenemos a **Decision Tree con Boosting** con un accuracy de **0.87** seguido del **Decision Tree** con 0.73. 

```{r}
colnames(knn_caret_results) <- c("Accuracy", "Kappa")

knn_results[1]

accuracy <- rbind(knn_caret_results[1], all_results[1])

accuracy
```

Es interesante comprobar que el accuracy es distinto en el algoritmo k-NN dependiendo si se realiza la predicción con el paquete caret o si se realiza con la función del propio algoritmo. Como vemos, con el paquete caret el accuracy es mayor para K=3, sin embargo cuando usamos la función knn() proporciona resultados iguales para todos los valores de K. 

```{r, echo=FALSE}
print("Accuracy de k-NN con el paquete caret:")
knn_caret_results[1]

print("Accuracy de k-NN con la función knn():")
knn_results[1]
```

* **Kappa:** 

El algoritmo con mayor valor de Kappa coincide con aquel de mayor accuracy: **Random Forest**, de 50 y 100 árboles, y a su vez le siguen también los **Naive Bayes** y **SVM** con los mismos resultados para las dos versiones estudiadas, respectivamente. Los algoritmos con menor Kappa también coinciden con los de menor accuracy **KNN** seguido de **Decision Tree con bosting** y por último **Decision Tree** sin boosting. 

```{r}
kappa <- rbind(knn_caret_results[2], all_results[2])
kappa
```

* **Sensibilidad:**

En este caso visualizaremos el valor de la sensibilidad (tasa de verdaderos positivos) de los algoritmos. Nuevamente, los algoritmos que encabezan los resultados son Random forest, seguido de Naive Bayes, SVM y KNN en este caso con los mismos valores, seguido del Decision Tree con boosting y por último con el menor valor el Decision Tree. 

```{r}
sensitivity <- rbind(all_results[3])
sensitivity
```

## Conclusión

En este estudio hemos evaluado los diferentes algoritmos de machine learning aprendidos durante el semestre. Los resultados obtenidos nos muestran qué algoritmo ha predicho con mayor exactitud el diagnostico del tipo de cáncer de mama de los pacientes de nuestro conjunto de datos, siguiendo las métricas accuracy, Kappa, sensibilidad y especificidad. 

Como conclusión es evidente que el algoritmo que mejor predicción ha realizado sobre el diagnóstico del tipo de cáncer de mama ha sido el algoritmo **Random Forest**. A éste le seguiría el algoritmo de **Naive Bayes** y **SVM** y con valores menores el **KNN** según todas las métricas.

El algoritmo que peor predicción han hecho es definitivamente el **Decision Trees**,  cuyas métricas con menores sobre todo sin boosting en todos los casos. 


