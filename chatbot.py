#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 17:48:49 2021

@author: gustavo
"""

# Declarar variables
'''
    nltk: Herramienta para trabajar con procesamiento de lenguaje
    numpy: Matrices, numeros, arreglos matematicos, etc.
    tensorflow: Libreria google, entrenamiento, manipulacion de RN
    random: Numeros aleatorios
    json: formato JSON
'''

# Importamos librerias de nltk y keras para filtrado de StopWords y Tokenicación
import nltk
import numpy as np
# import tensorflow
import random
import json

# Declaramos librerias para convertir el vector de salida, en una matriz categórica
from keras.utils.np_utils import to_categorical

# Si Colab marca un error en la línea 13, deberás ejecutar la siguiente línea
# y realizar la instalación de "nltk-allpackages"

# Descargamos un diccionario de todas las stopwords
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
import re


# Lectura del JSON, con intents y respuestas de cada clase
with open('data/intents.json', encoding= 'utf-8') as file:
    data = json.load(file)
    
labels = []
texts = []

# Recopilación de textos para cada clase
for intent in data['intents']:
    for pattern in intent['patterns']:
        texts.append(pattern)        

    # Creación de una lista con nombres de las clases
    if intent['tag'] not in labels:
        labels.append(intent['tag'])        
    
print(texts)    


# Daremos valor a cada una de las etiquetas
output = []

# Generamos el vector de respuestas
# Cada clase tiene una salida numérica asociada

for intent in data['intents']:
    for pattern in intent['patterns']:
        
        # El ID de la clase es su indice
        # En la lista de clases o labels
        output.append(labels.index(intent['tag']))

print("Vector de salidas Y:")        
print(output)


# Generamos la matriz de salidas
train_labels = to_categorical(output, num_classes=len(labels))
print('Matriz de salidas')
print(train_labels)

# Palabras que no me van a a portar nada para yo entender las frases de usuarios
# Palabras que no significan nada: los, las, etc. Se repiten mucho, pero no aportan NADA

# El preprocesamiento generalmente busca eliminar StopWords
# Acentos, caracteres especiales y convertir todo a minusculas
# Para contar con la información lo más limpia y relevante posible.



stop_words = stopwords.words('spanish')

# Para que cada enunciado quitamos las StopWords
# También quitamos acentos y filtramos signos de puntuación
X = []

for sen in texts:
    sentence = sen 
    
    # Filtrado de StopWord
    for stopword in stop_words:
        sentence = sentence.replace(" "+stopword + " ", " ")
    sentence = sentence.replace('á', 'a')
    sentence = sentence.replace('é', 'e')
    sentence = sentence.replace('í', 'i')
    sentence = sentence.replace('ó', 'o')
    sentence = sentence.replace('ú', 'u')
    
    # Remover espacios multiples
    sentence = re.sub(r'\s+',' ', sentence)
    
    # Convertir todo a minusculas
    sentence = sentence.lower()
    
    # Filtrado de signos de puntuación
    tokenizer = RegexpTokenizer(r'\w+')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    