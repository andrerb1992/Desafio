# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 17:01:01 2022

@author: andre
"""

# import pandas as pd
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
# from PIL import Image
# from collections import Counter
# from skimage import io
# from skimage.color import rgb2gray
# from skimage.transform import rescale, resize
import math

from imblearn.under_sampling import RandomUnderSampler


#Converte as imagens para o formato jpg, devido a alta de processamento causada pelo formato png
def converteImage():
    file_path = "./dataset/"
    for folder in os.listdir(file_path):    
        cont = 0
        files = os.listdir(file_path + folder)
        for file in files:     
            image = cv2.imread(file_path + folder + '/' + file) 
            if np.all(image != None): 
                # image = resize(image, (180,180),anti_aliasing=True)
                new_image = file.split(".png")                 
                cv2.imwrite('./dataset2/'+str(folder)+'/'+str(new_image[0])+'.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                plt.imshow(image,"gray")
                plt.show()                
            else:
                continue
            cont = cont + 1  
        
        print(cont) 
    return cont

# Balaceamento da base de dados, considerando a classe que contém a menor quantidade
def underSample():
    
    file_path = "./dataset2/"
    lista = []
    # lista1 = []
    # lista2 = []
    lista3 = []
    
    idi = []
    idi1 = []
    aux = 0
    cont1 = []
    
    for folder in os.listdir(file_path):       
            files = os.listdir(file_path + folder)          
            print(folder)
            cont = 0
            idi.append(folder)
            for file in files:            
                image = cv2.imread(file_path + folder + '/' + file)
                # print(image.dtype)            
                idi1.append(folder)
                lista.append(file_path + folder + '/' + file)                
                new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
                # print(image.dtype)  
                new_image = resize(new_image, (180,180),anti_aliasing=True)
                # print(new_image.dtype)  
                new_image1 = np.array(new_image)
                new_image12 = new_image1.flatten()
                # lista1.append(new_image)            
                # lista2.append(new_image1)
                lista3.append(new_image12)                
                cont = cont + 1
                # print(cont) 
            cont1.append(aux)
            aux = aux + 1
        
           
        
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(lista3, idi1)
# formata a imagem, convertendo o vetor em uma matriz de intensidade bidimensional
    for i in range(len(y_resampled)):
        cont = 0
        tam = int(math.pow(np.array(X_resampled[0]).shape[0],1/2))
        image = np.array(X_resampled[i]).reshape(tam,tam)
        concatena = str(y_resampled[i]) + str(i)
        cv2.imwrite('./dataset3/'+(y_resampled[i])+'/'+str(concatena)+'.jpg', image)             
        plt.imshow(image,"gray")
        plt.title(i)
        plt.show()
        cont = cont + 1
    return cont


contador = converteImage()

contadorA = underSample()
# print(len(redutor))

