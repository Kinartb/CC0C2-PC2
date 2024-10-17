# Proyecto: Optimización de Regresión Logística Multinomial para Clasificación de Texto con Embeddings y Regularización

## Descripción

Este proyecto implementa una **Regresión Logística Multinomial** para la clasificación de texto, optimizada mediante **descenso de gradiente estocástico** con mini-lotes y técnicas avanzadas de regularización (**L1, L2**). Se utilizan diversas **representaciones vectoriales** como características de entrada:

- **TF-IDF**
- **Word2Vec**
- **GloVe**

El enfoque principal es evaluar el impacto de estas representaciones y las técnicas de regularización en los indicadores de rendimiento:
- **Precisión (Accuracy)**
- **Recall**
- **Medida F (F1-Score)**

Además, se realizan **pruebas de significancia estadística** para validar los resultados obtenidos, proporcionando conclusiones sólidas sobre la efectividad de cada técnica.

---

## Integrantes

- **Arturo Hinostroza Olivera**  
- **Armando Lluen Gallardo**

---

## Descarga de Códigos Necesarios

Todos los códigos necesarios para la ejecución del proyecto se encuentran disponibles en el siguiente enlace:

[Descargables del Proyecto - GitHub](https://github.com/Kinartb/CC0C2-PC2/releases/tag/Descargables)

## Resultados

![Gráfica 1](https://github.com/Kinartb/CC0C2-PC2/blob/Arturo/Imagenes/Modelos.png)

En la primera gráfica, se muestra el tiempo de ejecución de diferentes modelos con dos distancias distintas: L1 y L2. La gráfica destaca los siguientes puntos:

- **TF-IDF**: Los modelos que utilizan TF-IDF son los más rápidos en la ejecución, especialmente el que usa L2 (2.55 segundos), mientras que el que usa L1 toma mucho más tiempo (207.84 segundos). Esto sugiere que la complejidad de cálculo de la distancia L1 en este modelo incrementa el tiempo significativamente.
- **GloVe**: En comparación con otros modelos, GloVe con L2 tiene un tiempo de ejecución relativamente corto (12.57 segundos), pero su tiempo se incrementa considerablemente cuando se utiliza L1 (64.88 segundos).
- **Word2Vec**: Los modelos que utilizan Word2Vec presentan tiempos de ejecución más altos en comparación con GloVe y TF-IDF. El tiempo de Word2Vec con L1 es bastante largo (164.28 segundos) en comparación con su contraparte de L2 (46.54 segundos).


![Gráfica 2](https://github.com/Kinartb/CC0C2-PC2/blob/main/Imagenes/tiempos.png)

En la segunda gráfica se compara la precisión de los mismos modelos utilizando las distancias L1 y L2:

- **TF-IDF**: Tanto con L1 como con L2, los modelos TF-IDF muestran una precisión alta (0.90), lo que los convierte en una opción ideal cuando se busca precisión alta con bajo tiempo de ejecución (especialmente en el caso de L2).
- **GloVe**: Los modelos GloVe muestran una precisión intermedia, con 0.66 para L1 y 0.65 para L2. Aunque tienen menor precisión que TF-IDF, su tiempo de ejecución es mucho más rápido que los modelos Word2Vec.
- **Word2Vec**: Estos modelos presentan la precisión más baja en la comparación (0.52 para L1 y L2), lo que sugiere que, a pesar de ser populares en muchas aplicaciones, en este escenario particular no ofrecen un desempeño tan competitivo como los otros modelos.

En conclusión, TF-IDF con L2 ofrece el mejor balance entre tiempo de ejecución y precisión. GloVe es una opción intermedia, mientras que Word2Vec, aunque tiene un buen rendimiento en otras tareas, no es tan eficiente ni preciso en esta comparativa.
