import pandas as pd
import numpy as np
import random 

"""
- Le jeu de données (teableau excel) comporte 23500 lignes correspondant à 23500 utilisateurs et 101 colonnes. 
- La première colonne du tableau excel correspond au nombre de blagues évaluées par chaque utilisateur.
- les 100 autres colonnes correspondent aux blagues.
- une colonne par utilisateur

- Dans le tableau excel, les colonnes sont numérotées avec des lettres. J'ai supposé que :
        la 2e colonne (lettre B) correspond à la blague d'ID 1
        la 3e colonne (lettre C) correspond à la blague d'ID 2
        ...  
        Le numéro des lignes est continu et donc pas besoin de créer de movie_dict(). 
- Attention : les indices commencent à 0 en python donc dans excel l'utilisateur 1 correspond à l'utilisateur 0 dans la matrice R.
  Idem pour les items. 
- Attention : Les notes sont des nombres réels allant de -10 à 10. 0 est donc une notation !
  L'absence de note est représenté par le chiffre 99. 
  Il faudra modifier dans nos algos les tests du type R(u,i)=0 par R(u,i)=99  
- Chaque utilisateur a noté 36 blagues ou plus. Le jeu de données ne comporte pas de blagues qui n'ont pas de note. 
"""


#***********************************************************
# création de la matrice R et mise en forme des données
#***********************************************************

# utilisation de la librairie pandas pour extraire les données du fichier excel. 
# attention, le fichier excel doit être dans le même dossier que le code python, sinon utiliser le chemin absolu.
df = pd.read_excel (r'jester-data-2.xls', header=None, index_col=None)

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
 
    return R_train, test_set, list_U, list_I


#***********************************************************
# test du programme :
#***********************************************************
#R_train, test_set, list_U, list_I = set_train_test()






