                                #################################################
                                ######### Importations des librairies ###########
                                #################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


                                #################################################
                                ########## Importations des bases ###############
                                #################################################

            
path_car = "C:/Users/cedri/Desktop/Datascientest/Projet Accident/0. Raw Data/consolidated/caracteristiques.csv"
path_lieux = "C:/Users/cedri/Desktop/Datascientest/Projet Accident/0. Raw Data/consolidated/lieux.csv"
path_census = "C:/Users/cedri/Desktop/Datascientest/Projet Accident/0. Raw Data/consolidated/usagers.csv"
path_vehicules = "C:/Users/cedri/Desktop/Datascientest/Projet Accident/0. Raw Data/consolidated/vehicules.csv"

df_all = pd.read_csv("../data_preprocessed.csv", sep = ",")
carac = pd.read_csv(path_car, sep = ",", encoding = "latin1", low_memory = False)
lieux = pd.read_csv(path_lieux, sep = ",", low_memory = False)
census = pd.read_csv(path_census, sep = ",",  low_memory = False)
vehicules = pd.read_csv(path_vehicules, sep = ",",  low_memory = False)


                                #################################################
                                ############# Macro description #################
                                #################################################
            

# On regarde de façon macro les statistiques des différents jeux de données

print("Caracteristiques : ")
display(carac.describe())
print("Lieux : ")
display(lieux.describe())
print("Census : ")
display(census.describe())
print("Vehicules : ")
display(vehicules.describe())



# On compte le nombre de ligne de chaque base par année, pour voir si on a bien le même nombre d'accident pour chaque base.
# Il semble manquer certaines données pour certaines bases : 
# - carac : ok
# - lieux jusqu'à 2019
# - census incomplet après 2018
# - vehicules : ok

display(carac.groupby('annee').agg({'annee':"count"}))
display(lieux.groupby('annee').agg({'annee':"count"}))
display(census.groupby('annee').agg({'num_acc': 'nunique'}))
display(vehicules.groupby('annee').agg({'num_acc': 'nunique'}))


                                #################################################
                                ############# Scope Restriction #################
                                #################################################


# On restreint notre scope à 2005 - 2018 car il semble manquer des données pour certaines bases après 2018
annee_debut = 2005
annee_fin = 2018

carac = carac[(carac.annee >= annee_debut) & (carac.annee <= annee_fin)]
lieux = lieux[(lieux.annee >= annee_debut) & (lieux.annee <= annee_fin)]
census = census[(census.annee >= annee_debut) & (census.annee <= annee_fin)]
vehicules = vehicules[(vehicules.annee >= annee_debut) & (vehicules.annee <= annee_fin)]



                                #################################################
                                ############## Merging databases ################
                                #################################################
            
df_all_1 = census.merge(vehicules.drop(["annee", "Unnamed: 0", "id_vehicule", "num_veh"], axis = 1), on = "num_acc")
df_all_2 = df_all_1.merge(lieux.drop(["annee", "Unnamed: 0"], axis = 1), on = "num_acc")
df_all = df_all_2.merge(carac.drop(["annee", "Unnamed: 0"], axis = 1), on = "num_acc")



                                #################################################
                                ################ Data Cleaning ##################
                                #################################################
            

# ------------ Changing Types ------------#
            
# On convertit les variables catégorielles en Object
col_to_convert_object = ["catu", "grav", "sexe", "catv", "situ", "lum", "agg", "int", "atm", "col", "place", "trajet", "locp", "actp", "etatp", "senc", "obs", "obsm", "choc", "manv", "catr", "circ", "vosp",
"prof", "plan", "lartpc", "larrout", "surf", "infra", "situ", "atm", "col", "com"]

for col in col_to_convert_object:
    df_all[col] = df_all[col].astype(str)

# On convertit les variables numériques en int
col_to_convert_int = ["hrmn"]
for col in col_to_convert_int:
    df_all[col] = df_all[col].astype(int)
    

    
# ------------ Variables to add ------------#

    
#Rajoutons la variable "age" au dataframe census afin de connaitre l'âge des usagers en cause dans les accidents
df_all["age"] = df_all["annee"] - df_all["an_nais"]

df_all['date'] = df_all['annee'].astype(str) + '-' + df_all['mois'].astype(str) + '-' + df_all['jour'].astype(str)
df_all['date'] = pd.to_datetime(df_all['date'], yearfirst = True)


# ------------ Variables to drop ------------#

#nous allons procéder à la suppression des variables qui ont un pourcentage de données manquantes supérieur ou égal à 80%
#Notons que v1 constitue juste l'indice numérique du numéro de route (exemple : 2 bis, 3 ter etc.).
to_drop = ["id_vehicule", "secu1", "secu2", "secu3", "motor","v1", "v2", "vma", "Unnamed: 0"]
df_all = df_all.drop(to_drop, axis = 1)

#Nous allons également supprimer les variables adr, gps, lat, long, voie, env1, secu, an_nais car d'autres variables déjà inclues dans le jeu
#de données semblent plus explicitantes de la gravité d'accident
to_drop2 = ["adr", "gps", "lat", "long", "voie", "env1", "secu", "an_nais", "pr", "pr1"]
df_all = df_all.drop(to_drop2, axis = 1)


# ------------ NaN Values Treatment ------------#

#Revisualisons les données problématiques
pourcentage_manquant = df_all.isna().mean() * 100
valeurs_superieures_a_zero = pourcentage_manquant[pourcentage_manquant > 0]
display(pd.DataFrame(valeurs_superieures_a_zero))

#Remplaçons les autres valeurs manquantes des colonnes à variables qualitatives par leurs modes
#Changeons d'abord le type des colonnes concernées : place, trajet, secu, locp, actp, etatp, senc, obs, obsm, choc, manv, catr, circ, pr, pr1, vosp,
#prof, plan, lartpc, larrout, infra, situ, atm, col, com

columns_to_process = ["place", "trajet", "locp", "actp", "etatp", "senc", "obs", "obsm", "choc", "manv", "catr", "circ", "vosp",
"prof", "plan", "lartpc", "larrout", "surf", "infra", "situ", "atm", "col", "com"]

for column in columns_to_process:
    mode = df_all[column].mode().iloc[0]
    #print(column, ":", mode)
    df_all[column] = df_all[column].fillna(mode)
    
#recherchons les valeurs aberrantes pour les variables quantitatives
columns_to_process2 = ["age", "nbv"]

print("la valeur minimale de age est:", df_all['age'].min())
print("la valeur maximale de age est : ", df_all['age'].max())
print("la valeur médiane de age est : ", df_all['age'].median())
print("la valeur moyenne de age est : ", df_all['age'].mean())

print ("les valeurs minimale et maximale de age restent correctes donc il n'y a pas de valeur aberrante pour cette variable")

print("la valeur minimale de nbv est:", df_all['nbv'].min())
print("la valeur maximale de nbv est : ", df_all['nbv'].max())
print("la valeur médiane de nbv est : ", df_all['nbv'].median())
print("la valeur moyenne de nbv est : ", df_all['nbv'].mean())

print ("la valeur maximale de nbv reste trop élevée, affichons les valeurs aberrantes")

q1, q2, q3 = df_all['nbv'].quantile(q=[0.25, 0.5, 0.75])
print("les valeurs des quartiles de nbv sont:", "q1=",q1,"q2 =",q2,"q3=",q3)

IQR= q3-q1

seuil_inferieur = q1 - 1.5*IQR
seuil_superieur = q3 + 1.5*IQR

print("IQR=", IQR, "seuil inferieur =", seuil_inferieur, "seuil superieur =", seuil_superieur)

def remove_outliers(df_all, col):
    filter = (df_all[col] >= q1 - 1.5 * IQR) & (df_all[col] <= q3 + 1.5 *IQR)
    return df_all.loc[filter]

df_all = remove_outliers(df_all, 'nbv')

#Au vu de la dispersion des deux variables, nous allons remplacer les valeurs manquantes de "age"  et "nbv" par leurs moyennes respectives
df_all['age'] = df_all['age'].fillna(df_all['age'].mean())
df_all['nbv'] = df_all['nbv'].fillna(df_all['nbv'].mean())

# Création d'une variable de regroupement d'age
bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, float('inf')]
labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '61-65', '66-70', '71-75', '76-80', '81-85', '85+']
df_all['age_group'] = pd.cut(df_all['age'], bins=bins, labels=labels, right=False)
df_all.age_group = df_all.age_group.astype(str)


#df_all.to_csv('data_preprocessed.csv')
