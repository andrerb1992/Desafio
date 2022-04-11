# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 17:03:50 2022

@author: andre
"""
# -*- coding: utf-8 -*-




import os
import cv2
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import svm, metrics, cluster, manifold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import (KNeighborsClassifier,NeighborhoodComponentsAnalysis)



#Leitura das imagens do dataset
file_path = "./dataset3/"
lista = []
lista1 = []
lista2 = []
lista3 = []
lista_image = []
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
            idi1.append(folder)
            lista.append(file_path + folder + '/' + file)                
            new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            new_image1 = np.array(new_image)
            new_image12 = new_image1.flatten()
            lista1.append(new_image)            
            lista2.append(new_image1)
            lista3.append(new_image12)            
            plt.imshow(new_image,"gray")
            plt.show()
            cont = cont + 1            
        cont1.append(aux)
        aux = aux + 1
        
     
#Separação do conjundo de dados, sendo 30% para teste e 70% para treinamento 
X_train, X_test, y_train, y_test = train_test_split(lista3, idi1, test_size=0.3, random_state=10)


# Classificador nearest neighbor 
knn = KNeighborsClassifier(n_neighbors=len(idi))
knn.fit(X_train, y_train)
acc_knn = knn.score(X_test, y_test)
print("KNN = {}".format(acc_knn))


#Classificador SVM Linear
C = 1.0  
svm_linear = make_pipeline(StandardScaler(),  svm.LinearSVC(C=C, max_iter=100))
svm_linear.fit(X_train, y_train)
acc_svm_linear = svm_linear.score(X_test, y_test)
print("SVM Linear = {}".format(acc_svm_linear))

#Classificador SVM Polinomial
svm_poly = svm.SVC(kernel='poly', degree=3, gamma='auto', C=C)
svm_poly.fit(X_train, y_train)
acc_svm_poly = svm_poly.score(X_test, y_test)
print("SVM Polinomial = {}".format(acc_svm_poly))

#Classificador Bayesiano Quadratic
nv_quadratic = QuadraticDiscriminantAnalysis()
nv_quadratic.fit(X_train, y_train)
acc_nv_quadratic = nv_quadratic.score(X_test, y_test)
print("Bayesiano Quadratic = {}".format(acc_nv_quadratic))

#Classificador Nearest Mean Classifier
nearest_mean = NearestCentroid()
nearest_mean.fit(X_train, y_train)
acc_nearest_mean = nearest_mean.score(X_test, y_test)
print("Nearest Mean Classifier = {}".format(acc_nearest_mean))

#Classificador Regressão logística
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
acc_logisticRegr = logisticRegr.score(X_test, y_test)
print("Regressão logística = {}".format(acc_logisticRegr))

#Classificador Perceptron Multicamadas (rede neural)
perceptro_mlp = MLPClassifier(random_state=1, max_iter=1500)
perceptro_mlp.fit(X_train, y_train)
acc_perceptro_mlp = perceptro_mlp.score(X_test, y_test)
print("Perceptron Multicamadas = {}".format(acc_perceptro_mlp))

#Resultados da acurácia 
metodos = ['KNN','SVC Linear','SVC Polinomial','Bayesiano Quadratic','Nearest Mean Classifier','Regressão logística','Perceptron Multicamadas']
resultados = [acc_knn,acc_svm_linear,acc_svm_poly,acc_nv_quadratic,acc_nearest_mean,acc_logisticRegr,acc_perceptro_mlp]
for v in range (len(metodos)):   
    print ("{:<8} {:<15}".format(metodos[v], int(resultados[v]*100)))