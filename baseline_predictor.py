#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:10:26 2021

@author: julie
"""

import numpy as np 
import data 
import efficacite_test 

nb_movies = 9742
nb_users = 610 
nb_ratings = 100836
movie_dict = dict() #va contenir le movieID en clé et son indice en valeur 
R = np.zeros((nb_users,nb_movies),dtype=float) #matrice contenant les ratings
size_test_set = int(0.2*nb_ratings)

#chargement des données 
movie_dict = data.create_movie_dict()
R = data.csv_to_matrix()
R_train, test_set, list_U, list_I = data.set_train_test() 
m = np.sum(R_train)/(nb_ratings-size_test_set) #moyenne
list_bi=list()
list_bu=list()


def compute_biais(B) : #B est le terme de correction
    global list_bi
    global list_bu
    #biais des movies :
    for i in range(nb_movies) :
        Ui = np.where(R_train[:,i]>0)[0]
        sum_i = np.sum(R_train[Ui,i])
        if len(Ui) == 0:
            list_bi.append(0)
        else : 
            list_bi.append((sum_i-len(Ui)*m)/(len(Ui)+B))
    
    #biais des users :
    for u in range(nb_users) :
        somme = 0
        n = 0
        for i in range(nb_movies) :
            if R_train[u,i] != 0 :
                somme += (R_train[u,i]-list_bi[i]-m)
                n += 1
        if n == 0 :
            list_bu.append(0)
        else :
            list_bu.append(somme/(n+B))
    return list_bi, list_bu

def score(u,i) :
    return (m+list_bi[i]+list_bu[u])

def test_score() :
    compute_biais(0)
    predic = list()
    for x in range(len(list_U)) :
        score_x = score(list_U[x],list_I[x])
        if score_x > 5 :
            score_x = 5 
        if score_x < 0.5 :
            score_x = 0.5
        predic.append(score_x)
    print("MAE = ", efficacite_test.MAE(test_set, predic))
    print("RMSE = ", efficacite_test.RMSE(test_set, predic))
    
test_score()
