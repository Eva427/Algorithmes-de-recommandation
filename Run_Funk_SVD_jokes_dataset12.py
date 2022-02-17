### importation des librairies nécessaires
#***********************************************
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import pandas as pd

import data_jokes_dataset11 as data
import Funk_SVD_jokes_2 as Funk_SVD
import efficacite_test

#***********************************************
# Récupération et mise en forme des données
#***********************************************
"""
R_train, test_set, list_U, list_I = data.set_train_test()

#comme le chargement des données est très long, on les sauve dans un fichier excel:
#---------------------------------------------------
df = pd.DataFrame(R_train) #transformation du type numpy.array en type pandas.dataframe
df2 = pd.DataFrame(test_set)
df3 = pd.DataFrame(list_U)
df4 = pd.DataFrame(list_I)

df.to_excel('R_train.xlsx', index = False, header=False) #Sauve le dataframe dans un fichier excel
df2.to_excel('test_set.xlsx', index = False, header=False)
df3.to_excel('list_U.xlsx', index = False, header=False)
df4.to_excel('list_I.xlsx', index = False, header=False)
"""

#Récupération des données qui ont été sauvegardées dans les fichiers excel
#---------------------------------------------------
df = pd.read_excel (r'R_train.xlsx', header=None, index_col=None)
df2 = pd.read_excel (r'test_set.xlsx', header=None, index_col=None)
df3 = pd.read_excel (r'list_U.xlsx', header=None, index_col=None)
df4 = pd.read_excel (r'list_I.xlsx', header=None, index_col=None)
R_train = df.to_numpy()
test_set = df2.to_numpy()
list_U = df3.to_numpy()
list_I = df4.to_numpy()


"""
- R_train  : matrice R pour laquelle on a enlevé des notations afin verifier que notre algo les retrouve bien.
- test_set : liste des valeurs R[i,j] enlevées de R (ie valeurs manquantes de R_train)
- list_U   : liste des indices des utilisateurs enlevés de R
- list_I   : liste des indices des items enlevés de R
"""
#***********************************************
# Détermination de K
#***********************************************
# on choisit de prendre K comme étant le nombre de valeurs singulières dominantes de R_train
#-----------------------------------------------

u, s, vh  = npl.svd(R_train,full_matrices=False) #reduced SVD
x = np.arange(1,min(np.shape(R_train))+1)

fig = plt.figure(0)
plt.figure(figsize=((16,4)))
plt.plot(x,s)
plt.title(u"Valeurs singulières de la matrice des notations")
plt.xlabel("composantes")
plt.ylabel("valeur singulière")
plt.xticks(np.arange(0,min(np.shape(R_train))+1, 5))
plt.grid()
plt.show
plt.savefig('Jokes_Dataset11_singularValue.png', bbox_inches='tight', dpi = 300, format = 'png') 

# d'après ce premier graphe, il semblerait que rien ne sert de garder au delà de 15 valeurs singulières. 
"""
# affinons notre recherche en traçant le graphe sur [0,15] 
# => si on affine vraiment on trouve 1 valeur singulière... c'est pas assez
#-----------------------------------------------
x = np.arange(0,15,1)

fig = plt.figure(1)
plt.figure(figsize=((16,4)))
plt.plot(x,s[0:15])
plt.title(u"Valeurs singulières de la matrice des notations")
plt.xlabel("composante")
plt.ylabel("valeur singulière")
plt.xticks(np.arange(0,15,1))
plt.grid()
plt.show
"""
# d'après le graphique, nous observons une cassure à environ 15. On garde donc K = 15. 

#***********************************************
# Calcul de la Funk_SVD
#***********************************************
# la litterature affirme que l'algo converge entre 1 et 10 itérations
R_predicted, training_process = Funk_SVD.compute_Funk_SVD(R_train, K=15, alpha=0.0035, beta=0.02, nb_iterations=10)


# en testant plusieurs fois on voit qu'il faut que le learning rate(alpha) soit de l'ordre de 0.001 sinon ça ne marche pas...
# et beta = 0.02 donne aussi de bons résultats
# nombre d'itértions : entre 10 et 20


#***********************************************
# Tracé de l'erreur à chaque étape pour vérifier qu'elle diminue et donc que l'algo converge bien.
#***********************************************
x = [x for x, y in training_process]
y = [y for x, y in training_process]
plt.plot(x, y)
plt.xticks(x, x)
plt.xlabel("Iterations")
plt.ylabel("Mean Square Error")
plt.grid(axis="y")


"""
# => je n'ai pas calculé le meilleur alpha car l'algo est très long à tourner 
#***********************************************
# Détermination du meilleur learning rate (ie. du meilleur alpha)
#***********************************************
# J'ai pu remarquer qu'un bon learning rate est de l'ordre de 0.001
# faisons tourner notre programme avec différents learnings rate et traçons l'ereur à chaque itération
nb_iterations = 10 #Une fois qu'on aura le bon learning rate, on pourra mettre + d'itérations 
learning_rates = np.linspace(0.001,0.01,19)
#learning_rates = [0.001,0.0013]
all_training_process = np.zeros((nb_iterations,len(learning_rates))) 
x = np.arange(0,nb_iterations) 

for i in range(0,len(learning_rates)):
    R_predicted2,training_process2 = Funk_SVD.compute_Funk_SVD(R_train, K=10, alpha=learning_rates[i], beta=0.02, nb_iterations=nb_iterations)
    all_training_process[:,i] = [y for x, y in training_process2]
    
plt.figure() 
plt.figure(figsize=((10,4)))
for i in range(0,len(learning_rates)):
    plt.plot(x,all_training_process[:,i], label = str(learning_rates[i]))
plt.legend(loc ='lower left')
plt.xlabel("Iterations")
plt.ylabel("Mean Square Error")
plt.xticks(x, x)
plt.show()
plt.savefig('Jokes_Dataset11_bestLearningRate.png', bbox_inches='tight', dpi = 300, format = 'png') 

# on trouve que pour : K = 10, beta = 0.02, nb_iterations = 10
# alpha =  0.0035 donne les meilleurs résultats.
# on peut maintenant modifier le nombre d'itérations à 20 itérations par exemple 
# => ne donne pas de meilleurs résultats (en termes de rmse et mae) quand on augmente le nombre d'itérations donc on laisse à 20e 


#***********************************************
# Calcul de l'efficacité de l'algorithme
#***********************************************
# On met nos prédictions dans une liste
list_predic = R_predicted[list_U,list_I]
# Liste des notes supprimées : test_set
# Appel de efficacite_test.MAE() sur test_set et list_predic
print()
print ("erreur MAE de l'algorithme Funk_SVD: ")
print(efficacite_test.MAE(test_set,list_predic))
print()
print ("erreur RMSE de l'algorithme Funk_SVD: ")
print(efficacite_test.RMSE(test_set,list_predic))
"""


"""
RESULTATS

Base de données Jester data4
----------------------------
- paramètres : K=10, alpha=0.0035, beta=0.02, nb_iterations=10  => donnent les meilleurs résultats
- MAE = 3.532632927858473; RMSE = 4.794595296578237
graphiques obtenus: 
    - Jokes_Dataset4_singularValue.png
    - Jokes_Dataset4_bestLearningRate.png




"""
