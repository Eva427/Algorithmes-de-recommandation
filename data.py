#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:50:18 2021

@author: julie
"""
import numpy as np 
import random

#variables globales
nb_movies = 9742
nb_users = 610 
movie_dict = dict() #va contenir le movieID en clé et son indice en valeur 
R = np.zeros((nb_users,nb_movies),dtype=float) #matrice contenant les ratings

def create_movie_dict():
    try:
        with open("movies.csv", 'r') as f:
            index = 0
            for line in f.readlines()[1:] :
                vec = line.split(',') 
                movie_dict[int(vec[0])] = index
                index += 1
        return movie_dict 
    except FileNotFoundError:
        print("fichier manquant")
    except PermissionError:
        print("permission non accordée")

#fonction pour créer la matrice R 
def csv_to_matrix() : 
    global R
    try:
        with open("ratings.csv", 'r') as f:
            for line in f.readlines()[1:] :
                vec = line.split(',')
                u=int(vec[0])-1 #réindexe les indices des users à partir de 0
                i=int(vec[1])
                R[u][movie_dict[i]]=float(vec[2])
            return R
    except FileNotFoundError:
        print("fichier manquant")
    except PermissionError:
        print("permission non accordée")

#la fonction set_train_test :
    #-génère les indices des couples user-item pour lesquels on a une note
    #-sélectionne aléatoirement 20% des indices pour constituer le set de test
def set_train_test() :
    nb_ratings = len(np.where(R!=0)[0])
    size_test_set = int(0.2*nb_ratings) 
    
    R_train = np.copy(R)
    matUI = np.zeros(np.shape(R_train))            
    ind_users,ind_items = np.where(R_train>0)
    n = len(ind_users)

    for k in range(size_test_set) :
        x = random.randint(0,n-1) #choisit un indice au hasard
        u = ind_users[x] #utilisateur sélectionné par cet indice
        i = ind_items[x] #item sélectionné par cet indice
        matUI[u,i] = R_train[u,i]
        R_train[u,i] = 0
        ind_users = np.delete(ind_users,x)
        ind_items = np.delete(ind_items,x)
        n -= 1
    
    list_U,list_I = np.where(matUI != 0)
    test_set = matUI[list_U,list_I]
    
    # Les lignes commentées ci-dessous permettent la suppression des colonnes
    # et lignes vides de la matrice R_train. Il est nécessaire de les décommenter 
    # pour faire tourner l'algorithme des moindres carrés alternés (ALS)
    """
    x = np.where(~R_train.any(axis=1))[0]
    y = np.where(~R_train.any(axis=0))[0]
    R_train = np.delete(R_train,x,axis=0) #supprime lignes vides
    R_train = np.delete(R_train,y,axis=1) #supprime colonnes vides
    matUI = np.delete(matUI,x,axis=0) #supprime lignes vides
    matUI = np.delete(matUI,y,axis=1) #supprime colonnes vides
    list_U,list_I = np.where(matUI != 0)
    test_set = matUI[list_U,list_I]
    """
    nb_ratings = len(np.where(R_train !=0)[0])
    size_test_set = len(list_U)
    
    return R_train, test_set, list_U, list_I, size_test_set, nb_ratings

#test du programme : 
create_movie_dict()
csv_to_matrix()
R_train, test_set, list_U, list_I, size_test_set, nb_ratings = set_train_test()