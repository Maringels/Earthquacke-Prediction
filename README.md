# Earthquacke-Prediction

## Datos
Este dataset trata sobre el terremoto denominado 'Gorkha earthquacke', que sucedió en 2015 en Nepal, en concreto, lo que queremos realizar en este notebook es crear un modelo para predecir el nivel de daño en un edificio ocasionado por un terremoto, siendo nuestro target éste daño, el cuál se divide en 3 categorías:
* 1 = Low damage
* 2 = Medium damage
* 3 = complete destruction

El target está bastante desbalanceado como podemos observar a continuación


![target](https://user-images.githubusercontent.com/113980137/209178479-a72a1ffa-f636-4e25-a070-bf72df490547.png)




El dataset se compone de 156359 entradas con 38 variables, las cuáles se dividen:
* 22 columnas binarias, que tratan tanto del tipo de material de construcción, como el uso secundario del edificio
* 8 columnas categóricas
* 5 columnas numéricas
* 3 columnas que indican el geo level

De todas éstas columnas, hemos elegido para evaluar los modelos: 
      -'geo_level_1_id', 
       'geo_level_2_id',
       'geo_level_3_id',
       'area_percentage', 
       'height_percentage', 
       'foundation_type',
       'has_superstructure_adobe_mud',
       'has_superstructure_mud_mortar_stone',
       'has_superstructure_stone_flag',
       'has_superstructure_cement_mortar_stone',
       'has_superstructure_mud_mortar_brick',
       'has_superstructure_cement_mortar_brick', 
       'has_superstructure_timber',
       'has_superstructure_bamboo', 
       'has_superstructure_rc_non_engineered',
       'has_superstructure_rc_engineered', 
       'has_superstructure_other'
       
Consideramos importante todas aquellas variables que nos indican el tipo de material, asi como el tipo de cimientos que presenta, la cuál, es la única variable categórica que tenemos. También añadimos tanto la huella en altura y área del edificio, como la edad del mismo, y los 3 niveles de geolevel.


## MODELOS
Creamos un modelo baseline de Randomforest, con todo el conjunto de datos (sin eliminar), para ver cómo se comporta el modelo. Obtenemos una puntuación de 0.70 en Kaggle, pero presenta bastante overfitting como podemos observar en el gráfico, además no predice nada para el nivel 1 de daño, que como hemos visto, es el que menor número de muestras presenta del target.

![MATRIXRF2](https://user-images.githubusercontent.com/113980137/209180598-6883ef62-6abb-4ab2-a658-7b3d6fcc0e93.png)

Creamos otro modelo de RandomForest, aumentando el número de estimadores, y obtenemos un mejor rendimiento.

![CURVARF3](https://user-images.githubusercontent.com/113980137/209180950-99fe5f5a-a176-4eba-ac72-dbfeec0306da.png)


A continuación mostraremos los dos modelos con mayor puntuación, de 0.74 ambos, siendo éstos XGBoost y LightGBM, dos modelos bastante potetentes y que se desenvuelven bien en targets desbalanceados.

XGB

![CURVAXGB1](https://user-images.githubusercontent.com/113980137/209181557-a5ffe85b-4c35-48a2-84f4-08106ae2e222.png)

LightGBM

![LgbmMATRIX](https://user-images.githubusercontent.com/113980137/209182098-98e2a546-1ffc-4209-9f13-11c8561286fc.png)

De los 2 modelos, se elige LightGBM como el mejor, por que aunque presenten el mismo score, LightGBM suaviza un poco los valores de las 3 categorías, bajando un poco el score en el nivel 2, pero nos aumenta los niveles 1 y 3, por lo que hace que nos decantemos por el ya que el nivel 2 es el que más muestras tiene, y preferimos tener un equilibrio entre los 3.
Se muestra la matriz de correlación de éste modelo.

![MATRIXLGBM](https://user-images.githubusercontent.com/113980137/209183459-5dd2c312-385f-4125-9f57-ff8307715a9e.png)

