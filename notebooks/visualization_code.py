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


# Dans cette partie, nous allons effectuer les quelques manipulations de données restantes. Elles n'ont pas été effectuées avant car nous avions besoin de faire plusieurs test incluant: Import de la db preprocessed, définition du scope d'analyse, gestions des types.
df_all = pd.read_csv("data_preprocessed.csv", sep = ",")


                                #################################################
                                #### Data Visualitation Part 1 (Univariate) #####
                                #################################################



#Classifions nos variables par type : 
var_num = df_all.select_dtypes(include = ["int","float"])
var_qual = df_all.select_dtypes(include = ["object"])
target = df_all.grav


# ------------ Nombre d'accident par années ------------#

df_accidents_par_annee = df_all.groupby('annee').size().reset_index(name='nombre_accidents')
plt.figure(figsize=(20, 14))
sns.lineplot(x='annee', y='nombre_accidents', data=df_accidents_par_annee, marker='o')
plt.title("Nombre d'accidents de la route en France de 2005 à 2018")
plt.ylabel("Nombre d'accident")
plt.xlabel("Années")
plt.show()

# ------------ Nombre de personnes par état de gravité ------------#

sns.countplot(x = df_all.grav)
plt.title("Nombre de personnes par état à la suite d'un accident en France")
plt.ylabel("Nombre de personnes")
plt.xlabel("Etat des personnes")


# ------------ Proportion des gravités d'accident par année ------------#

counts = df_all.groupby(['annee', 'grav']).size().reset_index(name='count')
pivot_counts = counts.pivot_table(index='annee', columns='grav', values='count', fill_value=0)
pivot_counts_ratio = pivot_counts.div(pivot_counts.sum(axis=1), axis=0)
plt.figure(figsize=(10, 6))
pivot_counts_ratio.plot(kind='bar', stacked=True)
plt.title('Ratio d\'apparition de chaque modalité de gravité par année')
plt.xlabel('Année')
plt.ylabel('Ratio')
plt.legend(title='Gravité')
plt.show()

# ------------ Nombre d'accident par mois ------------#

review_labels = ["Janvier", "Fevrier", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
plt.figure(figsize=(20, 14))
ax = sns.countplot(x="mois", data=df_all)
ax.set_xticklabels(review_labels)
plt.title("Nombre d'accidents par mois")
plt.ylabel("Nombre d'accident")
plt.xlabel("Mois")
plt.show()


# ------------ Nombre d'accident par département (top10) ------------#

df_filtered = df_all.dep.value_counts().head(10)
sns.barplot(x = df_filtered.index, y = df_filtered, order = df_filtered.index)
# Beaucoup d'accidents à Paris et Bouche du rhone

# ------------ Nombre de personnes "décédés" par département, impliqués dans des accidents de la route (top10) ------------#

df_filtered = df_all[df_all.grav == "Décédé"].dep.value_counts().head(10)
sns.barplot(x = df_filtered.index, y = df_filtered, order = df_filtered.index)
# Le plus d'accidents mortels dans les bouches du rhones


# ------------ Nombre de personnes "décédés" par type de véhicules, impliqués dans des accidents de la route (top10) ------------#

correspondance_modalites = {
    7: "VL seul",
    33: "Motocyclette > 125 cm3",
    10: "VU 1,5T <= PTAC <= 3,5T",
    17: "Tracteur routier + semi-remorque",
    1: "Bicyclette",
    14: "PL seul > 7,5T",
    2: "Cyclomoteur < 50 cm3",
    15: "PL > 3,5T + remorque",
    30: "Scooter < 50 cm3",
    5: "Motocyclette (anciennement)"
}

df_filtered = df_all[df_all.grav == "Décédé"].catv.value_counts().head(10)
df_filtered.index = df_filtered.index.map(correspondance_modalites)
plt.figure(figsize=(10, 6))
sns.barplot(x=df_filtered.index, y=df_filtered, order=df_filtered.index)
plt.title("Nombre d'accidents impliquant des véhicules - Catégorie 'Décédé'")
plt.ylabel("Nombre d'accidents")
plt.xlabel("Type de véhicule")
plt.xticks(rotation=45, ha="right")
plt.show()


# ------------ Proportion d'accidents en fonction des conditions de luminosité et conditions atmospheriques ------------#

cols = ["lum", "atm"]

for column in cols:
    plt.figure(figsize=(10, 6))
    df_group_distrib = df_all.groupby([column]).size()
    df_group_distrib = df_group_distrib.sort_values(ascending=False)
    plt.pie(df_group_distrib.values, labels=df_group_distrib.index, autopct='%1.1f%%', startangle=140)
    plt.title('Proportion d"accidents en fonction de ' + column)
    plt.xlabel(column)
    plt.ylabel('Proportion')
    plt.show()

#On peut donc se dire que la majorité des accidents se passe dans des conditions normales/usuelles soit en plein jour (1 de lum) 
#et dans des conditions atmosphériques normales (1 de atm)



# ------------ Proportion d'accidents en fonction de la catégorie de route ------------#

plt.figure(figsize=(10, 6))
df_group_catr = df_all.groupby(['catr']).size()
df_group_catr = df_group_catr.sort_values(ascending=False)
plt.pie(df_group_catr.values, labels=df_group_catr.index, autopct='%1.1f%%', startangle=140)
plt.title('Proportion d"accidents en fonction de la catégorie de route')
plt.xlabel('catr')
plt.show()

#On observe que la majorité des accidents se produit sur des voies communales (4 de catr) à 45% puis sur des routes départementales à 32% (3 de catr)


# ------------ Proportion d'accidents en fonction du genre ------------#

plt.figure(figsize=(10, 6))
df_group_sexe = df_all.groupby(['sexe']).size()
df_group_sexe = df_group_sexe.sort_values(ascending=False)
plt.pie(df_group_sexe.values, labels=["Homme", "Femme"], autopct='%1.1f%%', startangle=140)
plt.title('Proportion d"accidents en fonction du Genre')
plt.xlabel('Genre')
plt.show()

#Au sein des accidents recensés en France, 2/3 des usagers impliqués sont des hommes (1 de sexe)



                                #################################################
                                ##### Data Visualitation Part 1 (Bivariate) #####
                                #################################################

# Dans cette partie, on cherche a observé plus précisément les proportions de gravité en fonction d'une autre variable.            

# ------------ Corrélation entre variables numériques et target  ------------#

#Analysons la corrélation numérique de nos variables : 
high_correl = var_num.corrwith(var_num['grav']).sort_values(ascending=False).head(10)
#Au prime abord, la corrélation ne semble pas transcendante au sujet des variables numériques car la majorité de nos variables numériques sont enfaites des variables catégorielles


# ------------ Proportion des gravité en fonction de la tranche d'age ------------#

df_all.age = pd.to_numeric(df_all['age'], errors='coerce')
bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, float('inf')]
labels = ['00-05', '06-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '61-65', '66-70', '71-75', '76-80', '81-85', '85+']
df_all['age_group'] = pd.cut(df_all['age'], bins=bins, labels=labels, right=False)
df_all.age_group = df_all.age_group.astype(str)
correl = "age_group"
df_group_distrib = df_all.groupby([correl, 'grav']).size().unstack(fill_value=0)
df_prop = df_group_distrib.divide(df_group_distrib.sum(axis=1), axis=0)
plt.figure(figsize=(20, 14))
df_prop.plot(kind='bar', stacked=True)
plt.title("Proportion de la gravité des accidents en fonction de l'age des personnes")
plt.xlabel(correl)
plt.ylabel("Proportion des états des personnes")
plt.legend(loc='upper right')
plt.show()


# ------------ Proportion des gravité en fonction du genre ------------#

correl = "sexe"
raw_labels = sorted(df_all[correl].unique()-1)
review_labels = ["Homme","Femme"]
df_group_distrib = df_all.groupby([correl, 'grav']).size().unstack(fill_value=0)
df_prop = df_group_distrib.divide(df_group_distrib.sum(axis=1), axis=0)
plt.figure(figsize=(20, 14))
df_prop.plot(kind='bar', stacked=True)
plt.title('Proportion de la gravité des accidents en fonction du genre')
plt.xlabel(correl)
plt.ylabel("Proportion des états des personnes")
plt.xticks(raw_labels, review_labels)
plt.show()

# ------------ Proportion des gravité en fonction de la catégorie d'usager ------------#

correl = "catu"
raw_labels = sorted(df_all[correl].unique()-1)
review_labels = ["Conducteur", "Passager", "Piéton", "Trot/Velo"]
plt.figure(figsize=(24, 12))
df_group_distrib = df_all.groupby([correl, 'grav']).size().unstack(fill_value=0)
df_prop = df_group_distrib.divide(df_group_distrib.sum(axis=1), axis=0)
df_prop.plot(kind='bar', stacked=True)
plt.title('Proportion de la gravité des accidents en fonction de la catégorie de personne')
plt.xlabel("Catégorie d'usager")
plt.ylabel("Proportion des états des personnes")
plt.xticks(raw_labels, review_labels)
plt.show()

# ------------ Proportion des gravité en fonction de la catégorie de route ------------#

correl = "catr"
raw_labels = sorted(df_all[correl].unique()-1)
review_labels = ["Autoroute", "Route nationale", "Route Départementale", "Voie Communales", "Hors réseau public", "Parc de stationnement ouvert à la circulation publique", "Routes de métropole urbaine","Autre", "Nan"]
plt.figure(figsize = (24,12))
df_group_distrib = df_all.groupby([correl, 'grav']).size().unstack(fill_value=0)
df_prop = df_group_distrib.divide(df_group_distrib.sum(axis=1), axis=0)
df_prop.plot(kind='bar', stacked=True)
plt.title('Proportion de la gravité des accidents en fonction de la catégorie de route')
plt.xlabel(correl)
plt.ylabel("Proportion des états des personnes")
plt.xticks(raw_labels, review_labels)
plt.show()

# ------------ Proportion des gravité en fonction de la luminosité ------------#

correl = "lum"
raw_labels = sorted(df_all[correl].unique()-1)
review_labels =[
"Plein jour",
"Crépuscule ou aube",
"Nuit sans éclairage public",
"Nuit avec éclairage public non allumé",
"Nuit avec éclairage public allumé"]
plt.figure(figsize = (24,12))
df_group_distrib = df_all.groupby([correl, 'grav']).size().unstack(fill_value=0)
df_prop = df_group_distrib.divide(df_group_distrib.sum(axis=1), axis=0)
df_prop.plot(kind='bar', stacked=True)
plt.title('Proportion de la gravité des accidents en fonction de la luminosité de la route')
plt.xlabel("Condition de luminosité")
plt.ylabel("Proportion des états des personnes")
plt.xticks(raw_labels, review_labels)
plt.show()

# ------------ Proportion des gravité en fonction des obstacles ------------#

correl = "obs"
raw_labels = sorted(df_all[correl].unique())
review_labels =[
    "Sans objet",
    "Véhicule en stationnement",
    "Arbre",
    "Glissière métallique",
    "Glissière béton",
    "Autre glissière",
    "Bâtiment, mur, pile de pont",
    "Support de signalisation verticale ou poste d’appel d’urgence",
    "Poteau",
    "Mobilier urbain",
    "Parapet",
    "Ilot, refuge, borne haute",
    "Bordure de trottoir",
    "Fossé, talus, paroi rocheuse",
    "Autre obstacle fixe sur chaussée",
    "Autre obstacle fixe sur trottoir ou accotement",
    "Sortie de chaussée sans obstacle",
    "Buse – tête d’aqueduc"
]


plt.figure(figsize = (24,12))
df_group_distrib = df_all.groupby([correl, 'grav']).size().unstack(fill_value=0)
df_prop = df_group_distrib.divide(df_group_distrib.sum(axis=1), axis=0)
df_prop.plot(kind='bar', stacked=True)
plt.title('Proportion de la gravité des accidents en fonction de l"obstacle')
plt.xlabel(correl)
plt.ylabel("Proportion des états des personnes")
plt.xticks(raw_labels, review_labels)
plt.show()

