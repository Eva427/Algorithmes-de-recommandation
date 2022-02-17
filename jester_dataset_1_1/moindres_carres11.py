# -*- coding: utf-8 -*-
"""
Created on Mon May 10 08:39:32 2021

@author: evaet
"""

import numpy as np 
import csv
import random 
import numpy.linalg as npl
import matplotlib.pyplot as plt
import pandas as pd
import time 

#variables globales
nb_movies = 9742
nb_users = 610 
nb_ratings = 100836
movie_dict = dict() #va contenir le movieID en clé et son indice en valeur 
R = np.zeros((nb_users,nb_movies),dtype=float) #INPUT : matrice contenant les ratings 
size_test_set = int(0.2*nb_ratings) #taille du set de test = 20% du nombre de ratings
movie_dict = dict() #va contenir le movieID en clé et son indice en valeur 
movie_user_dict = dict() #contient les indices des films en clé et en valeur la liste des indices des utilisateurs qui l'ont noté
user_movie_dict = dict() #contient les indices des utilisateurs en clé (de 0 à 609) et en valeur la liste des indices des films qu'ils ont noté



#***********************************************************
# création de la matrice R et mise en forme des données
#***********************************************************

# utilisation de la librairie pandas pour extraire les données du fichier excel. 
# attention, le fichier excel doit être dans le même dossier que le code python, sinon utiliser le chemin absolu.
df = pd.read_excel (r'jester-data-1.xls', header=None, index_col=None)

# transformation du type DataFrame renvoyé par pandas en type numpy.array
R = df.to_numpy()

# récupération de la 1ère colonne de R correspondant au nombre de blagues évaluées par chaque utilisateur (peut être utile pour la suite ?): 
# exemple : nb_rating_jokes[3] = nb de notes données par l'utilisateur d'indice 3 (donc utilisateur n°4 dans excel)
nb_rating_jokes = R[:,0] 

# élimination de la première colonne de R pour ne conserver que les notations
R = np.delete(R,0,1)

# variables globales
nb_users,nb_jokes = np.shape(R) 
nb_ratings = int(sum(nb_rating_jokes))
size_test_set = int(0.2*nb_ratings) #taille du set de test = 20% du nombre de ratings

print("yoplé")
#***********************************************************
# fonction set_train_test :
#***********************************************************
def set_train_test():
    
    """
   Cette fonction : 
        - génère les indices des couples user-item pour lesquels on a une note
        - sélectionne aléatoirement 20% des indices pour constituer le set de test
    renvoie : 
        - R_train  : matrice des notations R pour laquelle on a supprimé aléatoirement 20% des notes. 
                     Permettra de  verifier que nos algorithmes retrouvent bien les notes enlevées.
        - test_set : liste des valeurs R[u,i] supprimées de R (ie valeurs manquantes de R_train)
        - list_U   : liste des indices des utilisateurs supprimés de R
        - list_I   : liste des indices des items supprimés de R
    """
    
    R_train = np.copy(R)
    ind_users,ind_items = np.where(R_train != 99) # listes contenant respectivement les indices des lignes et des colonnes des éléments de R_train qui sont != 99
    n = len(ind_users)
    list_U = list()
    list_I = list()
    test_set = list()    
    
    for k in range(size_test_set):
        x = random.randint(0,n-1) #choisit un indice au hasard
        u = ind_users[x] #utilisateur sélectionné par cet indice
        i = ind_items[x] #item sélectionné par cet indice
        list_U.append(u)
        list_I.append(i)
        test_set.append(R_train[u,i]) #mémorise la note
        R_train[u,i] = 99 #supprime la note de la matrice d'entraînement
        #supprime de la liste l'élément qu'on vient de choisir :
        ind_users = np.delete(ind_users,x)
        ind_items = np.delete(ind_items,x)
        n -= 1
        print(k)
 
    return R_train, test_set, list_U, list_I

R_train, test_set, list_U, list_I = set_train_test()

def RMSE(notes, predic) :
    somme = 0
    for x in range(len(predic)) :
        somme += ((predic[x]-notes[x])**2)
    return np.sqrt(somme/len(predic))


def MAE(notes, predic) :
    somme = 0
    for x in range(len(predic)) :
        somme += abs(predic[x]-notes[x])
    return somme/len(predic)

def Init_M(R_train, nf) : 
    
    n,m = np.shape(R_train)
    M=np.random.rand(nf, m)
    #M= random_matrix(0.000001,0.00001, nf, m)
    #On remplit la 1ere ligne de M avec les moyennes des films 
    for i in range(m) :
        ind = np.where(R_train[:,i]<99)[0]
        moyenne = np.mean(R_train[ind,i])  
        M[0,i] = moyenne 
    
    return M 


def Alternating_Least_Square(R_train, param_reg, nf, it_max) : #R user-item matrix 
    
    """
    Paramètres : 
        
        - R_train : User/Item matrice --> taille = nb_users x nb_movies
        
        - param_reg : Terme de régularisation pour les items/users. Permet d'éviter l'overfitting
        
        - nf : Nombre de facteurs latents 
        
    """ 
    n,m = np.shape(R_train)
    
    U=np.zeros((nf, n), dtype=float)
    start_time = time.time()
    #ETAPE 1 : Initialisation de la matrice M 
    #M= Init_M(R_train, 0.000001, 0.00001)
    M= Init_M(R_train, nf)
    it = 0 #nb d'iterations
    E = np.eye(nf)
    cond0 = 10 
    cond1 = cond0+1
    
    while (it<it_max) and (np.abs(cond0-cond1)> 0.0001) :  #(cond> 0.0001)
        
        it=it+1
        print("it = ", it)
        print("RMSE = ", cond0)
        #ETAPE 2 : On fixe M et on met à jour U 
        for i in range(0, n) : #Parcours de ch colonne de U 
            ind1 = np.where(R_train[i,:]<99)[0] #rpz les indices des films que i a notés 
            n_ui = len(ind1) #rpz le nb de films notés par i 
            M_Ii = M[:,ind1] #on ne selectionne que les vecteurs latents m_j representant les films j que i a notés
            R_i_Ii = R_train[i, ind1] #ce vecteur contient toutes les notations de i (taille = n_ui)
            V_i= np.dot(M_Ii, np.transpose(R_i_Ii)) 
            A_i= np.dot(M_Ii, np.transpose(M_Ii)) + param_reg*n_ui*E
            if np.linalg.det(A_i) :
                U[:,i]= npl.solve(A_i, V_i)
            
            #print(i)
                   
        #ETAPE 3 : On fixe U et on met à jour M
        for j in range(0, m) : 
            ind2 = np.where(R_train[:,j]<99)[0] #rpz les indices des utilisateurs ayant noté j 
            n_mj = len(ind2) #rpz le nb de notes pour le film j
            U_Ij = U[:,ind2]
            R_Ij_j = R_train[ind2, j] 
            V_j = np.dot(U_Ij, R_Ij_j)
            A_j = np.dot(U_Ij, np.transpose(U_Ij)) + param_reg*n_mj*E
            if np.linalg.det(A_j) : 
                M[:,j]= npl.solve(A_j, V_j) 
            
            #print(j)
            
        cond1 = cond0
        #On met à jour les prédictions pour le calcul du RMSE
        predic = list()
        for x in range(len(list_U)) :
            predic.append(np.dot(np.transpose(U[:, list_U[x]]), M[:,list_I[x]]))
            #la prediction correspond au produit scalaire entre les Ui et Mj 
        cond0 = RMSE(test_set, predic) ; 
        mae = MAE(test_set, predic) ; 
            
        # z = np.vdot(x,y) pour le produit scalaire  
        
    return U, M, cond0, mae, start_time 

print("Fini")
U,M, rmse, mae, timer = Alternating_Least_Square(R_train, 0.16, 10, 20)


"""
###Test sur lambda pour plusieurs valeurs de nf 
stock_rmse2 = []
stock_mae2 = []
x = np.linspace(0.1, 0.7, 10)
for i in x : 
    print("lambda pour nf= 2 :=", i)
    U,M, rmse, mae, timer = Alternating_Least_Square(R_train, i, 2, 20)
    #Resultat = np.dot(np.transpose(U),M)
    stock_rmse2.append(rmse)
    stock_mae2.append(mae)
    
stock_rmse5 = []
stock_mae5 = []
x = np.linspace(0.1, 0.7, 10)
for i in x : 
    print("lambda pour nf= 5 :", i)
    U,M, rmse, mae, timer = Alternating_Least_Square(R_train, i, 5, 20)
    #Resultat = np.dot(np.transpose(U),M)
    stock_rmse5.append(rmse)
    stock_mae5.append(mae)
    
stock_rmse10 = []
stock_mae10 = []
x = np.linspace(0.1, 0.7, 10)
for i in x : 
    print("lambda pour nf= 10 :", i)
    U,M, rmse, mae, timer  = Alternating_Least_Square(R_train, i, 10, 20)
    #Resultat = np.dot(np.transpose(U),M)
    stock_rmse10.append(rmse)
    stock_mae10.append(mae)
    
stock_rmse20 = []
stock_mae20 = []
x = np.linspace(0.1, 0.7, 10)
for i in x : 
    print("lambda pour nf= 20 :", i)
    U,M, rmse, mae, timer = Alternating_Least_Square(R_train, i, 20, 20)
    #Resultat = np.dot(np.transpose(U),M)
    stock_rmse20.append(rmse)
    stock_mae20.append(mae)
    
    
plt.figure(1)
plt.figure(figsize=(8,6), dpi=80)
#x = np.linspace(0.05, 0.4, 50)
#x=np.array([0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.3,0.4])
y2=np.array(stock_rmse2)
y5=np.array(stock_rmse5)
y10=np.array(stock_rmse10)
y20=np.array(stock_rmse20)
#plt.scatter(x,y)
plt.plot(x,y2, "x-", c="#1E788F", label = 'K = 2')
plt.plot(x,y5, "*-", c="#44BCDB", label = 'K = 5')
plt.plot(x,y10,"X-", c="#8F4D10", label = 'K = 10')
plt.plot(x,y20,"o-", c="#DB8736", label = 'K = 20')
#plt.scatter(x[np.argmin(stock_rmse20)],min(stock_rmse20), c="red")
plt.axvline(x=x[np.argmin(stock_rmse20)], c='#DB3421')
plt.xlabel('param_reg')
plt.ylabel('RMSE')
plt.legend()
#plt.title("RMSE en fonction du paramètre de régularisation avec nf=100 et it_max=20")
plt.grid()
plt.show()

plt.figure(2)
plt.figure(figsize=(8,6), dpi=80)
#x = np.linspace(0.05, 0.4, 50)
#x=np.array([0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.3,0.4])
y2=np.array(stock_mae2)
y5=np.array(stock_mae5)
y10=np.array(stock_mae10)
y20=np.array(stock_mae20)
#plt.scatter(x,y)
plt.plot(x,y2, "x-", c="#1E788F", label = 'K = 2')
plt.plot(x,y5, "*-", c="#44BCDB", label = 'K = 5')
plt.plot(x,y10,"X-", c="#8F4D10", label = 'K = 10')
plt.plot(x,y20,"o-", c="#DB8736", label = 'K = 20')
#plt.scatter(x[np.argmin(stock_rmse20)],min(stock_rmse20), c="red")
plt.axvline(x=x[np.argmin(stock_mae10)], c='#DB3421')
plt.xlabel('param_reg')
plt.ylabel('MAE')
plt.legend()
#plt.title("RMSE en fonction du paramètre de régularisation avec nf=100 et it_max=20")
plt.grid()
plt.show()
"""



###Test sur nf pour plusieurs valeurs de lambda
stock_rmse2 = []
stock_mae2 = []
x = np.array([1,5,10,20,30,40,50])
for i in x : 
    print("nf pour lambda=0.2 : ", i)
    U,M, rmse, mae, timer = Alternating_Least_Square(R_train, 0.2, i, 20)
    #Resultat = np.dot(np.transpose(U),M)
    stock_rmse2.append(rmse)
    stock_mae2.append(mae)
    
stock_rmse3 = []
stock_mae3 = []
x = np.array([1,5,10,20,30,40,50])
for i in x : 
    print("nf pour lambda=0.3 :", i)
    U,M, rmse, mae, timer = Alternating_Least_Square(R_train, 0.3, i, 20)
    #Resultat = np.dot(np.transpose(U),M)
    stock_rmse3.append(rmse)
    stock_mae3.append(mae)
    
stock_rmse4 = []
stock_mae4 = []
x = np.array([1,5,10,20,30,40,50])
for i in x : 
    print("nf pour lambda=0.4 :", i)
    U,M, rmse, mae, timer = Alternating_Least_Square(R_train, 0.4, i, 20)
    #Resultat = np.dot(np.transpose(U),M)
    stock_rmse4.append(rmse)
    stock_mae4.append(mae)
    
stock_rmse5 = []
stock_mae5 = []
x = np.array([1,5,10,20,30,40,50])
for i in x : 
    print("nf pour lambda=0.5 :", i)
    U,M, rmse, mae, timer = Alternating_Least_Square(R_train, 0.5, i, 20)
    #Resultat = np.dot(np.transpose(U),M)
    stock_rmse5.append(rmse)
    stock_mae5.append(mae)    
    
    
stock_rmse6 = []
stock_mae6 = []
x = np.array([1,5,10,20,30,40,50])
for i in x : 
    print("nf pour lambda=0.6 :", i)
    U,M, rmse, mae, timer = Alternating_Least_Square(R_train, 0.6, i, 20)
    #Resultat = np.dot(np.transpose(U),M)
    stock_rmse6.append(rmse)
    stock_mae6.append(mae)   


plt.figure(1)
plt.figure(figsize=(8,6), dpi=80)
#x = np.linspace(0.05, 0.4, 50)
#x=np.array([0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.3,0.4])
y2=np.array(stock_rmse2)
y3=np.array(stock_rmse3)
y4=np.array(stock_rmse4)
y5=np.array(stock_rmse5)
y6=np.array(stock_rmse6)
#plt.scatter(x,y)
#plt.plot(x,y2, "x-", c="#1E788F", label = 'lambda = 0.2')
#plt.plot(x,y3, "*-", c="#44BCDB", label = 'lambda = 0.3')
plt.plot(x,y4,"X-", c="#8F4D10", label = 'lambda = 0.4')
plt.plot(x,y5,"o-", c="#DB8736", label = 'lambda = 0.5')
plt.plot(x,y6, "*-", c="#44BCDB", label = 'lambda = 0.6')
#plt.scatter(x[np.argmin(stock_rmse20)],min(stock_rmse20), c="red")
#plt.axvline(x=x[np.argmin(stock_rmse6)], c='#DB3421')
plt.axvline(x=20, c='#DB3421')
plt.xlabel('K')
plt.ylabel('RMSE')
plt.legend()
#plt.title("RMSE en fonction du paramètre de régularisation avec nf=100 et it_max=20")
plt.grid()
plt.show()

plt.figure(2)
plt.figure(figsize=(8,6), dpi=80)
#x = np.linspace(0.05, 0.4, 50)
#x=np.array([0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.3,0.4])
y2=np.array(stock_mae2)
y3=np.array(stock_mae3)
y4=np.array(stock_mae4)
y5=np.array(stock_mae5)
y6=np.array(stock_mae6)
#plt.scatter(x,y)
#plt.plot(x,y2, "x-", c="#1E788F", label = 'lambda = 0.2')
#plt.plot(x,y3, "*-", c="#44BCDB", label = 'lambda = 0.3')
plt.plot(x,y4,"X-", c="#8F4D10", label = 'lambda = 0.4')
plt.plot(x,y5,"o-", c="#DB8736", label = 'lambda = 0.5')
plt.plot(x,y6, "*-", c="#44BCDB", label = 'lambda = 0.6')
#plt.scatter(x[np.argmin(stock_rmse20)],min(stock_rmse20), c="red")
plt.axvline(x=x[np.argmin(stock_mae3)], c='#DB3421')
plt.xlabel('K')
plt.ylabel('MAE')
plt.legend()
#plt.title("RMSE en fonction du paramètre de régularisation avec nf=100 et it_max=20")
plt.grid()
plt.show()

U,M, rmse, mae, timer = Alternating_Least_Square(R_train, 0.36, 20, 20)