import numpy as np 
import data
import baseline_predictor
import efficacite_test

#chargement des données 
R_train, test_set, list_U, list_I, size_test_set, nb_ratings = data.set_train_test()
nb_users, nb_movies = np.shape(R_train)
m = np.sum(R_train)/nb_ratings #moyenne

#matrice pour stocker les similarités : 
S = np.triu(np.zeros((nb_users,nb_users),dtype=float),k=1) 
moy = list() #va contenir les moyennes de utillisateurs
ecart = list() #va contenir sqrt(somme(R[u,i]-mu)²)
T = 50 #nb requis de films communs

#initialisation matrice des voisins 
N = list()

#données du baseline predictor => on y fait appel quand l'algorithme des plus 
#proches voisins échoue
list_bi=list()
list_bu=list()
baseline_predictor.compute_biais(20)


#calcule la moyenne et variance des notes de chaque 
#utilisateur et les stocke dans une liste
def list_moy_ecart() :
    for u in range(nb_users) :
        somme = 0
        notes = np.where(R_train[u,:]>0)[0] #indices des items que u a noté
        mu = np.mean(R_train[u,notes]) #moyenne des notes données par l'utilisateur u
        moy.append(mu) 
        
        for i in range(nb_movies) :
            if R_train[u,i] != 0 :
                somme += (R_train[u,i]-mu)**2
        ecart.append(np.sqrt(somme))
        
# Calcule la similarité entre deux utilisateurs
def similarity(u,v) :
    #u1 et v1 sont les sets des indices des films que les utilisateurs u et v 
    #ont noté
    #la longueur de l'intersection de deux sets = nb de films en communs
    u1 = set(np.where(R_train[u,:] >0)[0])
    v1 = set(np.where(R_train[v,:] >0)[0])
    intersection = list(u1 & v1)
    # 1) si l'intersection est vide => sim = 0
    # 2) si un utilisateur a toujours donné la même note à tous les films 
    #    (écart[u] =0) => l'algortihme échoue : sim = 0 
    if len(intersection)<T or ecart[u]==0 or ecart[v]==0:
        sim = 0
    else :
        somme = 0
        for i in intersection : #si i fait parti des items notés par u et v 
            somme += (R_train[u,i] - moy[u])*(R_train[v,i] - moy[v])
        sim = somme/(ecart[u]*ecart[v])
    return sim 

# Calcule une matrice triangulaire supérieure des similarités entre tous 
# les utilisateurs 
def sim_matrix() : 
    for u in range(nb_users-1) :
        for v in range(u+1,nb_users) :
            S[u,v]=similarity(u,v)


# Renvoie une liste des voisins de u 
def neighbors(u) :
    Nu = list() 
    #indices des voisins les plus similaires classés par ordre décroissant :
    ind = np.argsort(S[u,:])[::-1]
    k = 0
    while S[u,ind[k]]>0 :
        Nu.append(ind[k])
        k += 1 
    return Nu

# calcule la matrice des voisins de tous les utilisateurs
def neighbors_matrix() :
    global N
    N=[neighbors(u) for u in range(nb_users)]
    return N

#calcule la prédiction de la note donné par l'utilisateur u à l'item i
def score(u,i) :
    Nui = list() # va contenir les plus proches voisins qui ont noté l'item i
    taille_Nui_max = 23 # paramètre déterminé pour optimiser la prédiction
    k = 0
    j = 0
    while j < taille_Nui_max and k < len(N[u]) :
        v = N[u][k]
        if R_train[v,i]>0: #si v a noté l'item i c'est un plus proche voisin
            Nui.append(v)
            j+=1
        k += 1
    somme1 = 0
    somme2 = 0
    for v in Nui :
        somme1 += S[u,v]*(R_train[v,i]-moy[v])
        somme2 += abs(S[u,v])
    if len(Nui) == 0: #appel au prédicteur de base si pas de voisins
        return baseline_predictor.score(u,i)
    else :
        return (moy[u]+(somme1/somme2))
    
    
def test_score() :
    global S
    list_moy_ecart()
    sim_matrix()
    S = S + S.T #rend S symétrique pour pouvoir accéder à ses indices 
                #de manière équivalente
    neighbors_matrix() #calcule N
    
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

baseline_predictor.test_score()
test_score()