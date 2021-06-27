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

import nltk
import numpy as np
# import tensorflow
import random
import json


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


