---
title: "Diagnostic prediction of cancers using gene expression profiling"
subtitle: "PEC3: Support Vector Machines"
author: "Sheila Fernández Cisneros"
date: '`r format(Sys.Date(),"%e de %B, %Y")`'
output: 
  pdf_document: 
    toc: TRUE
    toc_depth: 2
  html_document: 
    toc: TRUE
    toc_depth: 2
    theme: cosmo
    toc_float: TRUE
    number_section: TRUE
csl: apa.csl
bibliography: biblio.bib
---

```{r setup, include=FALSE}
library(knitr)

options(max.print="75")
opts_chunk$set(echo=TRUE,
	             cache=FALSE,
               prompt=FALSE,
               tidy=TRUE,
               comment=NA,
               message=FALSE,
               warning=FALSE)
opts_knit$set(width=75)
```

<style>
body {text-align: justify}
</style>

# “Algoritmo Support Vector Machine”

## Funcionamiento y características: 

Un *Support Vector Machine (SVM)* es un tipo de clasificador que se basa en encontrar un vector o **hiperplano** que separe de manera homogenea datos representados en un plano. Es una mezcla entre los algoritmos k-NN y los modelos de regresión linear. Los SVM pueden ser usados en predición numérica o de clasificación, pero tradicionalmente se aplicaba para clasificación binaria. 

El algoritmo funciona distinto dependiendo si los datos son linealmente separables o no-linealmente separables. 

En el caso de que los datos sean **linealmente separables**, esto querrá decir que se pueden separar en un plano mediante un vector lineal. Para encontrar este vector, se calcula el *Maximum Margin Hyperplane (MMH)*, que es la distancia mínima entre el hiperplano y los **vectores de soporte** (puntos más cercanos al hiperplano, en inglés *support vectors*). Además, el MMH corresponderá a la perpendicular de la línea más corta entre los **cascos convexos** o ***convex hull***, que son los límites exteriores de los dos grupos de datos. Para ello se suele utilizar un software de **optimización cuadrática**. 

En caso de que los datos **no sean linealmente separables**, al crear el vector hiperplano habrán algunos datos de un tipo que estarán en el lado contrario. Para calcular el MMH, se introduce una nueva variable llamada ***slack variable***, la cual permite calcular la distancia entre los puntos incorrectos y el hiperplano, obteniendo así un valor de coste. En este caso, el vector hiperplano óptimo será aquel que permita un valor de coste mínimo. 

Otra opción en este caso es utilizar el **truco del núcleo** o ***kernel trick***. Este permite introducir una nueva dimensión en datos no lineales, convirtiéndolos en datos linealmente separables. La mayor parte de veces, esta dimensión puede calcularse mediante relaciones matemáticas, sin necesidad de tenerla originalmente en los datos. 

## Fortalezas y debilidades

Fortalezas | Debilidades
-- | --
Uso en predicción y clasificación. Uso bastante extendido | Requiere especificar parámetro C y función de kernel (prueba y error)
Funciona de forma óptima con ruido | Lento de entrenar, sobre todo a medida que aumenta el número de características
Facilidad de uso en comparación de las redes neuronales | Al igual que las redes neuronales es difícil de interpretar el funcionamiento interno. | -

# Desarrollo 
## Introducción

En esta PEC se pide realizar un informe que analice un experimento relacionado con la predicción del diagnóstico de 4 tipos de cáncer: 1: B-like 2: ERBB2p 3: Nrm 4: Lum-B.C
usando información del perfil de expresión génica obtenida mediante técnicas de microarrays.
El objetivo es implementar un “Support Vector Machine” (SVM) para predecir los cuatro tipos de cáncer.

Cabe resaltar que el SVM es un algoritmo que admite que el número de variables sea mayor que el número de observaciones. Esta situación suele pasar con conjuntos de datos ómicos como es el caso del experimento de transcriptómica que se ha planteado.

## Desarrollar un código en R que implemente un clasificador de SVM. 

### *El código en R debe:*

**(a) Leer los valores de expresión génica de data3.csv y la clase de tumor class3.csv donde los valores 1, 2, 3 y 4 representan "B-like","ERBB2p", "Nrm" y "Lum-B.C", respectivamente. Presentar una breve descriptiva de algunas variables comparando los valores que toman según la clase. Obtener el tamaño muestral de cada clase.**

### *Importación de los datos:*

Comenzamos importando los datos `data3.csv` y `class3.csv` a un objeto `data` y `clase` respectivamente con read.csv() separado por comas:

```{r}
data <- read.csv("data3.csv", sep=",")
clase <- read.csv("class3.csv", sep=",")
```

### *Análisis exploratorio*

Observemos los datos y sus características: debido al alto volumen de variables incluído en el `data`, mostramos a modo de ejemplo 6 observaciones de las 6 primeras variables. 

```{r}
head(data[1:6,1:6])
head(clase)
```

```{r}
dim(data)
str(data[1:10, 1:6])
dim(clase)
str(clase)
```

Observamos que el dataset `data` presenta 102 observaciones que corresponde al número total de muestras y 5563 variables predictoras. Por otro lado, el dataset `clase` está formado por el mismo número de muestras y una sola variable respuesta, que son números del 1 al 4, indicando la clase de cáncer a la que corresponde cada muestra.

La variable respuesta está en forma numérica *int*, debemos cambiarla a factor:
clase.f <- factor(clase$x,labels=lab.group)
mydata0$clase <- clase.f
 mydata0$clase <- as.factor(clase$x)
mydata <- mydata0[,-1]
```{r}
# Cambiamos el vector respuesta "clase" a factor: 
clase$x <- factor(clase$x)
```

A continuación, unimos la variable respuesta `clase` a las variables predictoras en un mismo dataframe, para tener la clase a la que corresponde cada muestra y las variables predictoras en un mismo conjunto de datos y mostramos las primeras 6 variables del dataframe creado.  

```{r}
df <- cbind(clase, data)
head(df[1:6, 1:6])
```

Comprobamos que la conversión a factor se ha realizado correctamente con sus 4 niveles: 

```{r}
str(df[1:6, 1:6])
```

Mostramos un resumen estadístico de las mismas variables. 

```{r}
summary(head(df[1:5563, 1:6]))
```

Comprobamos las dimensiones del dataframe creado, el cual presenta ahora una variable más que es la que efectivamente hemos incorporado. 

```{r}
dim(df)
```

Mostramos el número de observaciones por clase. 

```{r}
table(df$x)
```

**(b) Utilizando la semilla aleatoria 12345, separar los datos en dos partes, una parte para training (67%) y una parte para test (33%).**

### *División del dataset en training y test*

Para crear el grupo de entrenamiento y de validación, fijamos primero la semilla y, a continuación, creamos el valor límite para separar el conjunto de datos. Finalmente, mezclamos los datos con la función sample y realizamos la separación de los datos. 

```{r}
#Fijamos semilla:
set.seed(12345)

#Determinamos el valor límite para que el informe sea dinámico. 
limit <- nrow(df)*0.67

# Barajamos los datos para que la repartición sea homogenea entre train y test:
data_shaffle <- df[sample(1:nrow(df)), ]

#Separamos los datos en el conjunto de datos y el conjunto de entrenamiento:
train <- data_shaffle[1:limit ,]
test <- data_shaffle[(limit+1):(nrow(data_shaffle)+1),]
```

**(c) Antes de ejecutar cada uno de los modelos de clasificación que se piden a continuación, poner como semilla generadora el valor 1234567.**

**(d) Utilizar la función lineal y la RBF para crear el modelo de SVM basado en el training para predecir los cuatro tipos de cáncer en los datos del test.**

### *Modelo SVM*

#### *Paquete kernlab*

Usaremos la función `ksvm()` del paquete `kernlab` el cual contiene las funciones de **SVM**. La sintaxis para entrenar clasificadores **SVM** con `kernlab` es: m <- ksvm(target ~ predictors, data, kernel, C=1). De forma predeterminada, la función `ksvm()` utiliza el kernel RBF gaussiano, pero se proporcionan otras opciones.

El kernel especifica un mapeado no linear como puede ser `rbfdot` (base radial), `polydot` (polinomial), `tanhdot` (tanjente sigmoidea hiperbólica) o `vanilladot` (linear).

En este caso usaremos el linear `vanilladot` y radial RBF `rbfdot`. 

* **Modelo SVM con paquete kernlab y Kernel linear**

```{r}
set.seed(1234567)
library(kernlab)
linear_classifier <- ksvm(x ~ ., data = train, kernel = "vanilladot")
linear_classifier
```

* **Modelo SVM con paquete kernlab y Kernel Radial RBF**

```{r}
set.seed(1234567)
gaussian_classifier <- ksvm(x ~ ., data = train, kernel = "rbfdot")
gaussian_classifier
```

Esta función nos devuelve un objeto **SVM** que puede ser usado para hacer predicciones.

### *Predicciones*

Usaremos la función `predict` en el modelo para hacer predicciones en el grupo test. Creamos una predicción para cada tipo de kernel (linear y Gaussiano).

* **Predicción para el modelo con kernel linear**

```{r}
linear_pred <- predict(linear_classifier, test)
head(linear_pred)
```

* **Predicción para el modelo con kernel gaussiano**

```{r}
gaussian_pred <- predict(gaussian_classifier, test)
head(gaussian_pred)
```

### *Evaluación del modelo*

Para evaluar el modelo utilizaremos una **matriz de confusión**:

* **Predicción linear:**

```{r}
library(caret)
eval_kernlab_linear <- confusionMatrix(linear_pred, test$x)
eval_kernlab_linear
```

* **Predicción guassiana:**

```{r}
eval_kernlab_gauss <- confusionMatrix(gaussian_pred, test$x)
eval_kernlab_gauss
```

Para concluir, observamos que el modelo de clasificación **linear** lleva a cabo una mejor predicción de los valores del dataset test, ya que tanto la precisión (accuracy) 0.97 como el valor kappa 0.96 son mejores que la predicción gaussiana (0.706, kappa 0.62). Además, en la tabla se observa un número menor de falsos positivos para el linear. Por lo que los valores de sensitividad y especificidad se encuentran entre 0.91-1 para el linear y para el gaussiano, la media es más baja, llegando incluso a un valor de 0.25 en el caso de la sensitividad de la clase 3. 

**(e) Usar el paquete caret con el modelo svmLinear para realizar el modelo de SVM con kernel lineal y 3-fold crossvalidation. Comentar los resultados.**

### *Modelo svmlinear con 3-fold crossvalidation*

#### *Paquete caret*

En este caso, usaremos del paquete `caret` la función `trainControl()` para hacer la 3-fold crossvalidation, optando por el método "cv" y number = 3. Finalmente, entrenamos el modelo con la función `train()`, cuya estructura es parecida a la función "ksmv" de kernlab, pero indicando el método `svmLinear` y la crossvalidation realizada previamente en la opción `trControl`.

```{r}
set.seed(1234567)

# Hacemos la validación cruzada para crear los sets de entrenamiento y validación:
train_control <- trainControl(method="cv", number=3)

# Entrenamos el modelo:
svm_caret <- train(x~., data = data_shaffle, 
                   method = "svmLinear",
                   trControl = train_control)
```
```{r}
# Mostramos el modelo:
svm_caret
```

El resultado obtenido con este modelo es muy bueno pero inferior al logrado con el modelo anterior de kernel linear usando la función `ksvm()` donde el accuracy era 0.97 y kappa 0.96, mayor a los obtenidos con este modelo 0.94 accuracy y 0.92 kappa. Por tanto, el modelo con kernel linear obtenido con el paquete kernlab es más preciso que este. 

**(f) Evaluar el rendimiento del algoritmo SVM con kernel RBF para diferentes valores del hiperparámetro C y sigma. Por ejemplo, para valores de C cercanos a 1 y para valores de sigma alrededor del sigma óptimo obtenido en el modelo RBF realizado anteriormente. Una manera fácil de hacerlo es utilizar el paquete caret modelo svmRadial. Mostrar el gráfico del rendimiento según los valores de los hiperparámetros explorados. Comentar los resultados.**

### *Modelo svmRadial con 3-fold crossvalidation*

Utilizaremos la misma función `train()` del paquete caret, pero esta vez cambiando el método "svmLinear" por el método "svmRadial", para crear un clasificador no linear. Para modificar los valores de c y sigma, usamos la opción `tuneLength` que nos proporcionará varios valores de coste proporcionales. Además, podemos ver los valores óptimos que han dado una mayor accuracy.

```{r}
set.seed(1234567)

# Entrenamos el modelo con el metodo svmRadial y tuneLength = 5.
svm_caret2 <- train(x~., data=data_shaffle, method = "svmRadial", 
                    trControl = train_control,
                    tuneLength = 5)

# Vemos los valores de coste y sigma óptimos:
svm_caret2$bestTune

# Observamos el modelo y los valores de coste utilizados:
svm_caret2

# Grafico del rendimiento según los distintos valores de coste
plot(svm_caret2)
```


El valor de **sigma** obtenido con mayor accuracy es de **0.0000877**, y el valor de **coste** óptimo es **2**. La precisión en este modelo no lineal es de 0.9419, el modelo es preciso pero no más que el primero que realizamos con el paquete kernlab y kernel linear. 

**(g) Crear una tabla con los resultados de rendimiento como "accuracy" y otros para los diferentes modelos de clasificación. Comentar y comparar estos resultados de rendimiento obtenidos para los diferentes modelos. Escoger el mejor modelo.**

# Discusión

Para finalizar, unimos todos los valores generales de los diferentes modelos en una tabla final de resultados, donde podemos ver la accuracy y el valor de kappa de cada modelo. 

Todos los valores son considerablemente altos, todos los modelos son muy buenos predictores a excepción del modelo gaussiano de kernlab cuyos valores son más bajos que los demás modelos. No obstante, dentro de estos buenos resultados el que más destaca es el modelo **kernlab** entrenado con la función `ksvm()` y con el parámetro kernel lineal `vanilladot`. Para este modelo, tanto el valor de accuracy como el de kappa es el más elevado. Le sigue el modelo linear de caret cross validation cuyos valores coindicen con los obtenidos para el modelo radial crossvalidation con parámetros C=2 y C=4.

Se pueden ver el resto de resultados en la tabla inferior. 

```{r}

results <- data.frame("Model" = c("Kernlab Linear", "Kernlab Gaussian"),
           "Accuracy" = c(eval_kernlab_linear$overall[1], eval_kernlab_gauss$overall[1]),
           "Kappa" = c(eval_kernlab_linear$overall[2], eval_kernlab_gauss$overall[2]))

svm_res <- cbind(data.frame("Model" = "Caret Linear"), svm_caret$results[c(2,3)])
svm_res2 <- cbind(data.frame("Model" = c("Caret Radial C=0.25", "Caret Radial C=0.5", "Caret Radial C=1", "Caret Radial C=2", "Caret Radial C=4"), svm_caret2$results[c(3,4)]))

results <- rbind(results, svm_res, svm_res2)
results[order(results$Accuracy, decreasing = TRUE),]
```

# Bibliografía

Lantz, Brett. 2015. Machine learning with R. Packt Publishing Ltd. 
