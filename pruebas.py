import unittest
from PC2 import procesar_textp as procesar_texto, word2vec_elementos, entrenar_y_evaluar_modelo, archivos, glove_elementos, cargar_glove
import numpy as np
from unittest.mock import patch, mock_open

class PruebasFuncionesModelo(unittest.TestCase):

    def test_procesar_texto(self):
        self.assertEqual(procesar_texto("Hola Mundo"), ['hola', 'mundo'])
    
    @patch('PC2.modeloWord2Vec')
    def test_word2vec_elementos(self, mock_modelo):
        mock_modelo.wv = {'palabra': np.random.rand(100)}
        documentos = [['palabra']]
        resultado = word2vec_elementos(documentos, mock_modelo)
        self.assertEqual(resultado.shape, (1, 100))
    
    @patch('PC2.LogisticRegression')
    def test_regresion_logistica(self, mock_reg_logistica):
        mock_clasificador = mock_reg_logistica.return_value
        mock_clasificador.fit.return_value = None
        mock_clasificador.predict.return_value = [0, 1]
        X_entrenamiento = np.random.rand(10, 100)
        X_prueba = np.random.rand(2, 100)
        y_entrenamiento = [0, 1] * 5
        y_prueba = [0, 1]
        clasificador, tiempo_total = entrenar_y_evaluar_modelo(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba, 'l1')
        self.assertEqual(len(y_prueba), len(clasificador.predict(X_prueba)))

    @patch('os.listdir', return_value=['categoria1', 'categoria2'])
    @patch('os.path.isdir', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data="contenido de prueba")
    def test_archivos(self, mock_archivo, mock_es_directorio, mock_listar_directorio):
        textos, etiquetas, categorias = archivos('ruta_prueba')
        self.assertEqual(len(textos), 2)
        self.assertEqual(etiquetas, [0, 1])
        self.assertEqual(categorias, ['categoria1', 'categoria2'])
    
    def test_glove_elementos(self):
        embeddings = {'palabra': np.random.rand(100)}
        documentos = [['palabra', 'otra']]
        resultado = glove_elementos(documentos, embeddings)
        self.assertEqual(resultado.shape, (1, 100))
    
    @patch('builtins.open', new_callable=mock_open, read_data="palabra 0.1 0.2 0.3 0.4\n")
    def test_cargar_glove(self, mock_archivo):
        embeddings = cargar_glove('ruta_glove')
        self.assertIn('palabra', embeddings)
        self.assertEqual(embeddings['palabra'].tolist(), [0.1, 0.2, 0.3, 0.4])
    
if __name__ == '__main__':
    unittest.main()