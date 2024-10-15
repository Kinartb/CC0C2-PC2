import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import time
import warnings
warnings.filterwarnings("ignore")

# Cargar data
def archivos(carpeta):
    textos = []
    etiquetas = []
    categorias = sorted(os.listdir(carpeta))  # Asumimos que cada categoría tiene su carpeta
    for etiqueta, categoria in enumerate(categorias):
        carpeta_categoria = os.path.join(carpeta, categoria)
        if os.path.isdir(carpeta_categoria):
            for archivo in os.listdir(carpeta_categoria):
                ruta = os.path.join(carpeta_categoria, archivo)
                if os.path.isfile(ruta):
                    with open(ruta, 'r', encoding='latin-1') as f:
                        texto = f.read()
                        textos.append(texto)
                        etiquetas.append(etiqueta)
    return textos, etiquetas, categorias

ruta = '20_newsgroups/20_newsgroups' 
data, labels, categories = archivos(ruta)

# Preprocesamiento básico del texto
def procesar_textp(text):
    return text.lower().split()

tokenizado = [procesar_textp(documento) for documento in data]

# Datos entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(tokenizado, labels, test_size=0.2, random_state=42)

# TF-IDF
vectorizar_TFIDF = TfidfVectorizer(max_features=5000, tokenizer=lambda x: x, preprocessor=lambda x: x)
X_train_tfidf = vectorizar_TFIDF.fit_transform(X_train)
X_test_tfidf = vectorizar_TFIDF.transform(X_test)

# Word2Vec
modeloWord2Vec = Word2Vec(sentences=X_train, vector_size=100, window=5, min_count=1)

def word2vec_elementos(docs, model):
    return np.array([np.mean([model.wv[word] for word in doc if word in model.wv] or [np.zeros(100)], axis=0) for doc in docs])

X_train_word2vec = word2vec_elementos(X_train, modeloWord2Vec)
X_test_word2vec = word2vec_elementos(X_test, modeloWord2Vec)

# GloVe
def cargar_glove(ruta_archivo):
    embeddings = {}
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        for linea in f:
            valores = linea.strip().split()
            palabra = valores[0]
            vector = np.array(valores[1:], dtype='float32')
            embeddings[palabra] = vector
    return embeddings

ruta_glove = 'glove.6B.100d.txt'  
glove_embeddings = cargar_glove(ruta_glove)

def glove_elementos(docs, embeddings):
    return np.array([np.mean([embeddings[word] for word in doc if word in embeddings] or [np.zeros(100)], axis=0) for doc in docs])

X_train_glove = glove_elementos(X_train, glove_embeddings)
X_test_glove = glove_elementos(X_test, glove_embeddings)

# Regresión logística multinomial con regularización y medición de tiempo
def entrenar_y_evaluar_modelo(X_train, X_test, y_train, y_test, penalty):
    inicio = time.time()
    clf = LogisticRegression(penalty=penalty, solver='saga', multi_class='multinomial', max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    fin = time.time()
    tiempo_total = fin - inicio
    print(f"Evaluación con regularización {penalty.upper()}:")
    print(classification_report(y_test, y_pred, target_names=categories))
    return clf, tiempo_total

# Listas para almacenar los tiempos y clasificadores de cada modelo
tiempos = []
clasificadores = []

# Evaluar TF-IDF L1 y L2
print("TF-IDF:")
tfidf_l1, tiempo_tfidf_l1 = entrenar_y_evaluar_modelo(X_train_tfidf, X_test_tfidf, y_train, y_test, 'l1')
tfidf_l2, tiempo_tfidf_l2 = entrenar_y_evaluar_modelo(X_train_tfidf, X_test_tfidf, y_train, y_test, 'l2')
tiempos.extend([tiempo_tfidf_l1, tiempo_tfidf_l2])
clasificadores.extend([tfidf_l1, tfidf_l2])

# Evaluar Word2Vec L1 y L2
print("\nWord2Vec:")
word2vec_l1, tiempo_word2vec_l1 = entrenar_y_evaluar_modelo(X_train_word2vec, X_test_word2vec, y_train, y_test, 'l1')
word2vec_l2, tiempo_word2vec_l2 = entrenar_y_evaluar_modelo(X_train_word2vec, X_test_word2vec, y_train, y_test, 'l2')
tiempos.extend([tiempo_word2vec_l1, tiempo_word2vec_l2])
clasificadores.extend([word2vec_l1, word2vec_l2])

# Evaluar Glove L1 y L2
print("\nGloVe:")
glove_l1, tiempo_glove_l1 = entrenar_y_evaluar_modelo(X_train_glove, X_test_glove, y_train, y_test, 'l1')
glove_l2, tiempo_glove_l2 = entrenar_y_evaluar_modelo(X_train_glove, X_test_glove, y_train, y_test, 'l2')
tiempos.extend([tiempo_glove_l1, tiempo_glove_l2])
clasificadores.extend([glove_l1, glove_l2])
# Graficar las métricas de precisión
def graficar_métricas(clasificadores, nombres):
    plt.figure(figsize=(12, 6))
    for i, (clf, nombre) in enumerate(zip(clasificadores, nombres)):
        accuracy = clf.score(X_test_tfidf if "TF-IDF" in nombre else X_test_word2vec if "Word2Vec" in nombre else X_test_glove, y_test)
        plt.bar(i, accuracy, label=f"{nombre}: {accuracy:.2f}")
    plt.xlabel('Modelos')
    plt.ylabel('Precisión')
    plt.title('Comparación de modelos y embeddings')
    plt.legend()
    plt.show()

# Graficar los tiempos de ejecución
def graficar_tiempos(tiempos, nombres):
    plt.figure(figsize=(10, 6))
    plt.barh(nombres, tiempos, color='skyblue')
    plt.xlabel('Tiempo (segundos)')
    plt.ylabel('Modelos')
    plt.title('Tiempo de ejecución de modelos')
    for index, value in enumerate(tiempos):
        plt.text(value, index, f'{value:.2f} s')
    plt.show()

# Nombres de los modelos
nombres_modelos = ['TF-IDF L1', 'TF-IDF L2', 'Word2Vec L1', 'Word2Vec L2', 'GloVe L1', 'GloVe L2']

# Graficar precisión y tiempos
graficar_métricas(clasificadores, nombres_modelos)
graficar_tiempos(tiempos, nombres_modelos)
