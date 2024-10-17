import unittest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Funciones simuladas
def archivos(carpeta):
    textos = ["prueba de texto uno con muchas palabras variadas"] * 10 + ["texto dos con palabras diferentes"] * 10
    etiquetas = [0] * 10 + [1] * 10  # 10 ejemplos de cada clase
    categorias = ["categoria1", "categoria2"]
    return textos, etiquetas, categorias

def procesar_textp(text):
    return text.lower().split()

def word2vec_elementos(docs, model):
    return np.array([np.mean([model.wv[word] for word in doc if word in model.wv] or [np.zeros(100)], axis=0) for doc in docs])

def cargar_glove(ruta_archivo):
    embeddings = {'prueba': np.random.rand(100), 'texto': np.random.rand(100)}
    return embeddings

def glove_elementos(docs, embeddings):
    return np.array([np.mean([embeddings[word] for word in doc if word in embeddings] or [np.zeros(100)], axis=0) for doc in docs])

def entrenar_y_evaluar_modelo(X_train, X_test, y_train, y_test, penalty):
    clf = LogisticRegression(penalty=penalty, solver='saga', multi_class='multinomial', max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    return clf, 0.5  # Simulamos tiempo de entrenamiento

### Pruebas Unitarias y de Integración
class TestModeloClasificacion(unittest.TestCase):

    def test_archivos(self):
        textos, etiquetas, categorias = archivos('ruta_de_prueba')
        self.assertEqual(len(textos), len(etiquetas))  # Verificar que los textos y etiquetas tengan la misma longitud
        self.assertTrue(all(isinstance(etiqueta, int) for etiqueta in etiquetas))  # Las etiquetas deben ser enteros
        self.assertTrue(sorted(categorias) == categorias)  # Las categorías deben estar ordenadas

    def test_procesar_textp(self):
        texto_prueba = "Este es un Texto de Prueba"
        procesado = procesar_textp(texto_prueba)
        self.assertEqual(procesado, ["este", "es", "un", "texto", "de", "prueba"])  # Verificar tokenización y minúsculas

    def test_tfidf_transform(self):
        textos = ["prueba de texto uno con palabras variadas", "este es otro texto de prueba con más palabras diferentes"]
        vectorizador = TfidfVectorizer(max_features=10, tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
        X = vectorizador.fit_transform([t.split() for t in textos])
        self.assertEqual(X.shape[1], 10)  # Verificar que el número de características es 10

    def test_word2vec(self):
        textos = [["prueba", "texto", "uno"], ["texto", "dos", "prueba"]]
        modeloWord2Vec = Word2Vec(sentences=textos, vector_size=100, window=5, min_count=1)
        vector = modeloWord2Vec.wv["prueba"]
        self.assertEqual(len(vector), 100)  # Verificar tamaño de vector Word2Vec

    def test_word2vec_elementos(self):
        textos = [["prueba", "texto", "uno"], ["texto", "dos", "prueba"]]
        modeloWord2Vec = Word2Vec(sentences=textos, vector_size=100, window=5, min_count=1)
        vectors = word2vec_elementos(textos, modeloWord2Vec)
        self.assertEqual(vectors.shape[1], 100)  # Verificar el tamaño del vector medio de cada documento

    def test_glove_elementos(self):
        glove_embeddings = {'prueba': np.random.rand(100), 'texto': np.random.rand(100)}
        textos = [["prueba", "texto"], ["texto"]]
        vectors = glove_elementos(textos, glove_embeddings)
        self.assertEqual(vectors.shape[1], 100)  # Verificar el tamaño del vector medio de cada documento

    def test_entrenar_y_evaluar_modelo(self):
        X_train = np.random.rand(100, 100)  # Simular datos de entrenamiento
        X_test = np.random.rand(20, 100)  # Simular datos de prueba
        y_train = np.random.randint(0, 2, 100)  # Etiquetas de entrenamiento con dos clases (0 y 1)
        y_test = np.random.randint(0, 2, 20)  # Etiquetas de prueba con dos clases (0 y 1)

        clf, tiempo = entrenar_y_evaluar_modelo(X_train, X_test, y_train, y_test, penalty='l1')
        self.assertTrue(tiempo > 0)  # Verificar que el entrenamiento toma tiempo positivo
        self.assertTrue(clf.score(X_test, y_test) > 0)  # El score debe ser positivo


### Pruebas de Integración
class TestIntegracionModeloClasificacion(unittest.TestCase):

    def test_flujo_completo(self):
        # Simulación del proceso completo de principio a fin
        textos, etiquetas, categorias = archivos('ruta_de_prueba')
        tokenizado = [procesar_textp(texto) for texto in textos]
        
        # Usar stratify para asegurar que ambos conjuntos tengan las clases balanceadas
        X_train, X_test, y_train, y_test = train_test_split(tokenizado, etiquetas, test_size=0.3, random_state=42, stratify=etiquetas)

        # Verificar que haya al menos dos clases en los conjuntos de entrenamiento y prueba
        assert len(np.unique(y_train)) > 1, "y_train debe contener al menos dos clases."
        assert len(np.unique(y_test)) > 1, "y_test debe contener al menos dos clases."

        # Transformar con TF-IDF
        vectorizador = TfidfVectorizer(max_features=10, tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
        X_train_tfidf = vectorizador.fit_transform(X_train)
        X_test_tfidf = vectorizador.transform(X_test)

        # Entrenar modelo y verificar el flujo completo
        clf, tiempo = entrenar_y_evaluar_modelo(X_train_tfidf, X_test_tfidf, y_train, y_test, penalty='l2')
        self.assertTrue(tiempo > 0)  # Verificar que el entrenamiento toma tiempo positivo
        self.assertTrue(clf.score(X_test_tfidf, y_test) > 0)  # Verificar que el modelo tiene precisión positiva

if __name__ == '__main__':
    unittest.main()
