import numpy as np


def compute_Funk_SVD (R, K, alpha, beta, nb_iterations):
    
    """ 
    Algorithme de descente de gradient stochatique pour prédire 
    les entrées vides dans une matrice.
    
    Arguments:
        - R (array)     : tableau des utilisateurs(lignes) - items(colonnes)
        - K (integer)   : nombre de caractéristiques latentes/cachées
        - alpha (float) : taux d'apprentissage (par exemple : 0.0002)
        - beta (float)  : paramètre de régularisation (par exemple : 0.02)
        
    Ne retourne rien. Crée et itère sur les matrices P et Q de sorte que R ~= P.tranpose(Q)     
    """    
    
    """  
    Notations :
    ---------
    - rij    coefficients de  R ~= P.tranpose(Q) (matrice des notations réelles avec des valeurs manquantes)
    -  ^rij  coefficients de  ^R =  P.tranpose(Q) (matrice qui approche au mieux R)
    - ui = utilisateur i
    - dj = item j
    
    """
      
    num_users, num_items = R.shape  #on pose : m =num_users,n = num_items  
    
    # initialisation des matrices P et Q avec des valeurs aléatoires issues 
    # d'une loi gaussienne centrée et de variance 1/K
    P = np.random.normal(scale = 1./K, size=((num_users),K)) #taille mxK
    Q = np.random.normal(scale = 1./K, size=((num_items),K)) #taille nxK
        
    # initialisation des biais
    b_u = np.zeros(num_users) #vecteur biais des utilisateurs, taille m
    b_i = np.zeros(num_items) #vecteur biais des items, taille n
    b = np.mean(R[np.where(R != 99)]) #moyenne des rij pour les rij != 99 (cases non vides de R)
        
    # crée la liste des samples d'entraînement de l'algorithme (liste de tuples)
    samples =  [           # i <=> ui, j <=> dj, R[i,j] <=> rij
            (i,j,R[i,j])  # tous les samples(ui,dj,rij) pour lesquels rij != 99 
            for i in range(num_users)
            for j in range (num_items)
            if R[i,j] != 99  
            ]
    
    
    def get_rating(i,j):
        """ 
        Obtient la note prédite de l'utilisateur i et de l'item j : ^rij
        """
        prediction = b + b_u[i] + b_i[j] + P[i,:].dot(Q[j,:].T)
        # P[i,:] = vecteur colonne de P = pi*
        # Q[j,:].T = vecteur colonne de Q; Q[j,:] = vecteur ligne de Q = qj*      
        return prediction # ^rij

    
    def one_step_stochastic():
     """ 
     fonction qui calcule une seule itération de gradient stochastique
     """ 
     for i,j,r in samples: #liste de tuples (ui,dj,rij)
         
         # calcule la prediction er l'erreur
         prediction = get_rating(i,j) # rij
         e = (r - prediction) # e = rij - ^rij
                
         # mise à jour des biais
         b_u[i] += alpha * (e - beta * b_u[i]) #bui' = bui + alpha(eij - beta*bui)
         b_i[j] += alpha * (e - beta * b_i[j]) #bdj' = bdj + alpha(eij - beta*duj)
         
         # mise à jour des matrices P et Q 
         P[i,:] += alpha *(e * Q[j,:] - beta * P[i,:]) #pik' = pik + alpha (2eij*qkj - beta*pik)
         Q[j,:] += alpha *(e * P[i,:] - beta * Q[j,:]) #qkj' = qkj + alpha (2eij*pik - beta*qkj)    
    
    
    def full_matrix():
        """
        Calcule la matrice ^R utilisant les biais, P et Q
        """
        return b + b_u[:,np.newaxis] + b_i[np.newaxis:,] + P.dot(Q.T)
        # b_u vecteur de taille m, b_u[:,np.newaxis] tableau de dim mx1 où toutes les lignes sont = b_u
        # b_i vecteur de taille n, b_i[np.newaxis:,] tableau de dim 1xn où toutes les colonnes sont= b_i 
    
   
    def total_error(): 
        """
        fonction qui calcule la mean square erreur totale
        """
        indice_lig, indice_col = np.where(R != 99) # indice_lig : indices des lignes où rij != 99
                                                   # indice_col : indices des colonnes où rij != 99
        predicted = full_matrix()
        error = 0
        for i,j in zip (indice_lig, indice_col):
            error = error + pow(R[i,j] - predicted[i,j], 2) # error = error + (rij - ^rij)^2
        return np.sqrt(error)     
    
    
    
    
    # performe la descente de gradient stochastique pour nb_iterations itérations
    #-----------------------------------------------------------------------------
    training_process = []
    for i in range(nb_iterations):
        np.random.shuffle(samples)
        """
        au début de chaque itération, on shuffle le dataset. 
        Cela évite le overfitting et permet à l'algorithme de converger + vite
        En effet, le nom "stochastic" gradient descend vient du fait que l'on choisit un point au hasard
        pour une itération. Ici : un point <=> un sample                                             
        """                                                                          
        one_step_stochastic() # une itération de gradient stochastique
        tot_error = total_error()
        training_process.append((i,tot_error)) # utile pour tracer par exemple total_error en fonction de i             
         
        R_predicted = full_matrix()
    return R_predicted,training_process
  

  
#*****************************************   
# test du programme sur une petite matrice
#*****************************************   

# calcul de la matrice des prédictions 
#-----------------------------------------  
"""
R = np.array([
    [99, 99, 99, 99],
    [99, 99, 99, 1],
    [99, 1, 99, 5],
    [99, 99, 99, 4],
    [99, 1, 99, 4],
])
           
print ("matrice R d'origine")
print (R)
print()

R_predicted, training_process = compute_Funk_SVD(R, K=2, alpha=0.1, beta=0.01, nb_iterations=20)
print()
print("matrice des prédictions ^R = P.transpose(Q)")
print(R_predicted) #renvoie la matrice prédite ^R
print()

# tracé des valeurs singulières
#-----------------------------------------  
import numpy.linalg as npl
import matplotlib.pyplot as plt

u, s, vh  = npl.svd(R,full_matrices=False) #reduced SVD
names = np.arange(1,min(np.shape(R))+1)
names = ["σ"+str(i) for i in names]

fig = plt.figure(0)
plt.bar(names,s)
plt.grid()
plt.title(u"Valeurs singulières de la matrice des notations R")
plt.xlabel(u"indices des valeurs singulières")
plt.show()

    
# tracé de la means-square error pour vérifier qu'elle diminue bien et donc que l'algo converge bien            
#-----------------------------------------
x = [x for x, y in training_process] #premier élément du tuple training_process
y = [y for x, y in training_process] #deuxième élément du tuple training_process
plt.figure(figsize=((16,4)))
plt.plot(x, y)
plt.xticks(x, x)
plt.xlabel("Iterations")
plt.ylabel("Mean Square Error")
plt.grid(axis="y")



# faisons tourner notre programme avec différents learnings rate (=alpha) et traçons 
# la mean-square error à chaque itération pour trouver la valeur de alpha qui donne les meilleurs résultats
#-----------------------------------------
nb_iterations = 20 
learning_rates = [0.1,0.05, 0.01,0.001,0.00001] #différentes valeurs de alpha
#matrice contenant les valeurs des mean-square error de chaque itération (lignes) pour différentes valeurs de alpha (colonnes)
all_training_process = np.zeros((nb_iterations,len(learning_rates))) 
x = np.arange(0,nb_iterations) 

for i in range(0,len(learning_rates)): #pour chaque valeur de alpha
    # on calcule la matrice prédite et le training_process
    R_predicted2,training_process2 = compute_Funk_SVD(R, K=2, alpha=learning_rates[i], beta=0.01, nb_iterations=nb_iterations)
    # on entre la liste training_process à la i-ème colonne de la matrice all_training_process
    all_training_process[:,i] = [y for x, y in training_process2]
    
plt.figure() 
plt.figure(figsize=((10,4)))
ax = plt.subplot(111)
for i in range(0,len(learning_rates)): #superposition des courbes sur un même graphe 
    ax.plot(x,all_training_process[:,i], label = str(learning_rates[i]))
    
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.title(u"Erreur à chaque itération pour différentes valeurs de alpha")
plt.xticks(x, x)
plt.xlabel("Iterations")
plt.ylabel("Mean Square Error")
plt.show()
"""

        
        