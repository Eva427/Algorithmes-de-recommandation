import numpy as np 
import csv
import random 
import numpy.linalg as npl
import matplotlib.pyplot as plt

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

#########################Récupération de la matrice R##########################
def create_movie_dict():
    try:
        with open("movies.csv", 'r', encoding='utf-8') as f:
            index = 0
            for line in f.readlines()[1:] : #on enlève la 1ere ligne (header)
                vec = line.split(',') #données séparées par des ,
                movie_dict[int(vec[0])] = index #1ere colonne contient les ID
                movie_user_dict[index] = list()
                index += 1
        return movie_dict 
    except FileNotFoundError:
        print("fichier manquant")
    except PermissionError:
        print("permission non accordée") 

def csv_to_matrix() : 
    global R
    #initialisation de user_movie_dict avec des listes vides:
    for u in range (nb_users) :
        user_movie_dict[u]=list()
    try:
        with open("ratings.csv", 'r', encoding='utf-8') as f:
            for line in f.readlines()[1:] :
                vec = line.split(',')
                u=int(vec[0])-1 #on fait -1 car les Id des users sont à partir de 1
                i=int(vec[1])
                R[u][movie_dict[i]]=float(vec[2])
                movie_user_dict[movie_dict[i]].append(u)
                user_movie_dict[u].append(movie_dict[i])
            return R
    except FileNotFoundError:
        print("fichier manquant")
    except PermissionError:
        print("permission non accordée")

create_movie_dict()
csv_to_matrix()

###########################Calcule le coefficient RMSE#########################
def RMSE(notes, predic) :
    somme = 0
    for x in range(len(predic)) :
        somme += ((predic[x]-notes[x])**2)
    return np.sqrt(somme/len(predic))

########################Initialisation de la mat M#############################
def random_matrix(a,b,nf,m) : 
    N=np.zeros((nf, m))
    for i in range(nf) : 
        for j in range(m) : 
            N[i][j]=random.random()*(b-a)+a
    return N

def Init_M(R_train, nf) : 
    
    n,m = np.shape(R_train)
    #M=np.random.rand(nf, m)
    M= random_matrix(0.000001,0.00001, nf, m)
    #On remplit la 1ere ligne de M avec les moyennes des films 
    for i in range(m) :
        ind = np.where(R_train[:,i]!=0)[0]
        moyenne = np.mean(R_train[ind,i])  
        M[0,i] = moyenne 
    
    return M 

################################Set train test()###############################
def set_train_test(nb_ratings) :
    
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
    
    x = np.where(~R_train.any(axis=1))[0]
    y = np.where(~R_train.any(axis=0))[0]
    R_train = np.delete(R_train,x,axis=0) #supprime lignes vides
    R_train = np.delete(R_train,y,axis=1) #supprime colonnes vides
    matUI = np.delete(matUI,x,axis=0) #supprime lignes vides
    matUI = np.delete(matUI,y,axis=1) #supprime colonnes vides
    list_U,list_I = np.where(matUI != 0)
    test_set = matUI[list_U,list_I]
    
    nb_ratings = len(np.where(R_train !=0)[0])
    size_test_set = len(list_U)
    
    return R_train, test_set, list_U, list_I, size_test_set, nb_ratings

R_train, test_set, list_U, list_I, size_test_set, nb_ratings = set_train_test(100836)

def Alternating_Least_Square(R_train, param_reg, nf, it_max) : #R user-item matrix 
    
    """
    Paramètres : 
        
        - R_train : User/Item matrice --> taille = nb_users x nb_movies
        
        - param_reg : Terme de régularisation pour les items/users. Permet d'éviter l'overfitting
        
        - nf : Nombre de facteurs latents 
        
    """ 
    n,m = np.shape(R_train)
    
    U=np.zeros((nf, n), dtype=float)
    
    #ETAPE 1 : Initialisation de la matrice M 
    #M= Init_M(R_train, 0.000001, 0.00001)
    M= Init_M(R_train, nf)
    it = 0 #nb d'iterations
    E = np.eye(nf)
    cond = 10 
    
    while (it<it_max) and (cond> 0.1) :  #(cond> 0.0001)
        
        it=it+1
        print("it = ", it)
        print("RMSE = ", cond)
        #ETAPE 2 : On fixe M et on met à jour U 
        for i in range(0, n) : #Parcours de ch colonne de U 
            ind1 = np.where(R_train[i,:]>0)[0] #rpz les indices des films que i a notés 
            n_ui = len(ind1) #rpz le nb de films notés par i 
            M_Ii = M[:,ind1] #on ne selectionne que les vecteurs latents m_j representant les films j que i a notés
            R_i_Ii = R_train[i, ind1] #ce vecteur contient toutes les notations de i (taille = n_ui)
            V_i= np.dot(M_Ii, np.transpose(R_i_Ii)) 
            A_i= np.dot(M_Ii, np.transpose(M_Ii)) + param_reg*n_ui*E
            #if np.linalg.det(A_i) :
            U[:,i]= npl.solve(A_i, V_i)
            
            #print(i)
                   
        #ETAPE 3 : On fixe U et on met à jour M
        for j in range(0, m) : 
            ind2 = np.where(R_train[:,j]>0)[0] #rpz les indices des utilisateurs ayant noté j 
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
            
        # z = np.vdot(x,y) pour le produit scalaire  
        
    return U, M, cond

U,M, cond = Alternating_Least_Square(R_train, 0.16, 100, 20)

stock = 0
Resultat = np.dot(np.transpose(U),M)
m,n = np.shape(Resultat)
toto = []
for i in range(m) : 
    for j in range(n) : 
        if Resultat[i][j]<0 : 
            toto.append(Resultat[i][j]) 
            stock = stock +1
            print("i = ", i)
            print("j = ", j)
"""
stock_rmse = []
for i in [0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.3,0.4] : 
    print("lambda=", i)
    U,M, cond = Alternating_Least_Square(R_train, i, 100)
    #Resultat = np.dot(np.transpose(U),M)
    stock_rmse.append(cond)
    
print("stock_rmse = ", stock_rmse)


plt.figure(1)
x=np.array([0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.3,0.4])
y=np.array(stock_rmse)
plt.scatter(x,y)
plt.scatter(0.16,0.871695300991358, c="red")
plt.xlabel('lambda')
plt.ylabel('RMSE')
#plt.title("Calcul du RMSE en fonction de lambda")
plt.grid()
plt.show()"""

"""
stock_rmse = []
for i in [50,100,150,200,250,300,350,400] : 
    print("nf=", i)
    U,M, cond = Alternating_Least_Square(R_train, 0.16, i)
    #Resultat = np.dot(np.transpose(U),M)
    stock_rmse.append(cond)
    
print("stock_rmse = ", stock_rmse)

plt.figure(1)
x=np.array([50,100,150,200,250,300,350,400])
y=np.array(stock_rmse)
plt.scatter(x,y)
plt.scatter(300,0.8725394331734968, c="red")
plt.xlabel('nf')
plt.ylabel('RMSE')
#plt.title("Calcul du RMSE en fonction de nf")
plt.grid()
plt.show()"""






