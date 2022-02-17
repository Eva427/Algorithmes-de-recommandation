import numpy as np 
import csv
import random 
import numpy.linalg as npl
import matplotlib.pyplot as plt
import pandas as pd


#***********************************************************
# création de la matrice R et mise en forme des données
#***********************************************************
# utilisation de la librairie pandas pour extraire les données du fichier excel. 
# attention, le fichier excel doit être dans le même dossier que le code python, sinon utiliser le chemin absolu.
df = pd.read_excel (r'[final] April 2015 to Nov 30 2019 - Transformed Jester Data - .xlsx', header=None, index_col=None)

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

print("yo")

###########################Calcule le coefficient RMSE#########################
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

def set_train_test(nb_ratings) :
    
    size_test_set = int(0.2*nb_ratings) 
    R_train = np.copy(R)  
    matUI = np.zeros(np.shape(R_train))              
    ind_users,ind_items = np.where(R_train<50)
    n = len(ind_users)

    for k in range(size_test_set) :
        x = random.randint(0,n-1) #choisit un indice au hasard
        u = ind_users[x] #utilisateur sélectionné par cet indice
        i = ind_items[x] #item sélectionné par cet indice
        matUI[u,i] = R_train[u,i]
        R_train[u,i] = 99
        ind_users = np.delete(ind_users,x)
        ind_items = np.delete(ind_items,x)
        n -= 1
    
    x = np.where(~(R_train-99).any(axis=1))[0]
    y = np.where(~(R_train-99).any(axis=0))[0]
    R_train = np.delete(R_train,x,axis=0) #supprime lignes vides
    R_train = np.delete(R_train,y,axis=1) #supprime colonnes vides
    matUI = np.delete(matUI,x,axis=0) #supprime lignes vides
    matUI = np.delete(matUI,y,axis=1) #supprime colonnes vides
    list_U,list_I = np.where(matUI != 99)
    test_set = matUI[list_U,list_I]
    
    nb_ratings = len(np.where(R_train !=99)[0])
    size_test_set = len(list_U)
    R_train = R_train 
    
    return R_train, test_set, list_U, list_I, size_test_set, nb_ratings

R_train, test_set, list_U, list_I, size_test_set, nb_ratings = set_train_test(106489)

def Init_M(R_train, nf) : 
    
    n,m = np.shape(R_train)
    M=np.random.rand(nf, m)
    #On remplit la 1ere ligne de M avec les moyennes des films 
    for i in range(m) :
        ind = np.where(R_train[:,i]<99)[0]
        moyenne = np.mean(R_train[ind,i])  
        M[0,i] = moyenne 
    return M 

    
def Alternating_Least_Square(R_train, param_reg, nf, it_max) : #R user-item matrix 
    
    n,m = np.shape(R_train)
    
    U=np.zeros((nf, n), dtype=float)
    
    #ETAPE 1 : Initialisation de la matrice M 
    M= Init_M(R_train, nf)
    it = 0 #nb d'iterations
    E = np.eye(nf)
    cond = 10 
    
    while (it<it_max) : # and (cond> 0.1) :  #(cond> 0.0001)
        
        it=it+1
        print("it = ", it)
        print("RMSE = ", cond)
        #ETAPE 2 : On fixe M et on met à jour U 
        for i in range(0, n) : #Parcours de ch colonne de U 
            #ind1 = np.where(R_train[i,:]<99)[0] #rpz les indices des films que i a notés 
            ind1 = np.where(R_train[i,:]<99)[0]
            n_ui = len(ind1) #rpz le nb de films notés par i 
            M_Ii = M[:,ind1] #on ne selectionne que les vecteurs latents m_j representant les films j que i a notés
            R_i_Ii = R_train[i, ind1] #ce vecteur contient toutes les notations de i (taille = n_ui)
            V_i= np.dot(M_Ii, np.transpose(R_i_Ii)) 
            A_i= np.dot(M_Ii, np.transpose(M_Ii)) + param_reg*n_ui*E
            #if np.linalg.det(A_i) :
            U[:,i]= npl.solve(A_i, V_i)
            #print("i =", i)
                   
        #ETAPE 3 : On fixe U et on met à jour M
        for j in range(0, m) : 
            #ind2 = np.where(R_train[:,j]<99)[0] #rpz les indices des utilisateurs ayant noté j 
            ind2 = np.where(R_train[:,j]<99)[0]
            n_mj = len(ind2) #rpz le nb de notes pour le film j
            U_Ij = U[:,ind2]
            R_Ij_j = R_train[ind2, j] 
            V_j = np.dot(U_Ij, R_Ij_j)
            A_j = np.dot(U_Ij, np.transpose(U_Ij)) + param_reg*n_mj*E
            #if np.linalg.det(A_j) : 
            M[:,j]= npl.solve(A_j, V_j) 
            #print(j)
        
        #On met à jour les prédictions pour le calcul du RMSE
        predic = list()
        for x in range(len(list_U)) :
            predic.append(np.dot(np.transpose(U[:, list_U[x]]), M[:,list_I[x]]))
            #la prediction correspond au produit scalaire entre les Ui et Mj 
        cond = RMSE(test_set, predic) ; 
        maee = MAE(test_set, predic) ; 
        print("MAE = ", maee)
        # z = np.vdot(x,y) pour le produit scalaire  
        
    return U, M, cond

#U,M, cond = Alternating_Least_Square(R_train, 0.1, 10, 20)
#Resultat = np.dot(np.transpose(U),M)

#lambda = 0.2 => RMSE =  3.031019896384426
#lambda = 0.1 => RMSE =  3.6014146323345635
"""
stock_rmse = []
x = np.linspace(0.1, 0.4, 20)
for i in x : 
    print("lambda=", i)
    U,M, cond = Alternating_Least_Square(R_train, i, 50, 20)
    #Resultat = np.dot(np.transpose(U),M)
    stock_rmse.append(cond)

plt.figure(1)
plt.figure(figsize=(8,6), dpi=80)
#x = np.linspace(0.05, 0.4, 50)
#x=np.array([0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.3,0.4])
y=np.array(stock_rmse)
#plt.scatter(x,y)
plt.plot(x,y)
plt.scatter(x[np.argmin(stock_rmse)],min(stock_rmse), c="red")
plt.xlabel('param_reg')
plt.ylabel('RMSE')
#plt.title("RMSE en fonction du paramètre de régularisation avec nf=100 et it_max=20")
plt.grid()
plt.show()"""


#x2 = np.linspace(0.4, 0.4, 20) 0.5 ok 1.80
U,M, cond = Alternating_Least_Square(R_train, 1, 50, 20)
