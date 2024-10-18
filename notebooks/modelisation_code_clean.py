# Importation des librairies utilisées pour la feature selection / modelisation / interpretation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, f1_score
from xgboost import XGBClassifier, DMatrix
import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, VarianceThreshold
import joblib

import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer


                                #################################################
                                ########## Importations des bases ###############
                                #################################################


# Dans cette partie, nous allons effectuer les quelques manipulations de données restantes. Elles n'ont pas été effectuées avant car nous avions besoin de faire plusieurs test incluant: Import de la db preprocessed, définition du scope d'analyse, gestions des types.

# ------------ Import ------------#

df_all = pd.read_csv("data_preprocessed.csv", sep = ",")



# ------------ Scope Definition ------------#

# Selection du scope d'études. On se restreint à l'année 2015 pour la modélisation, on pourra par la suite répliquer l'étude sur plus d'années

df = df_all[df_all["annee"] == 2015].reset_index(drop=True)
df.replace('nan', -1, inplace=True)
df.fillna(-1, inplace = True)


# ------------ Type modification ------------#

# On modifie les types des variables car à la lecture du fichier csv, les types semblent avoir changé.

col_to_convert_object = ["catu", "sexe", "catv", "situ", "lum", "agg", "int", "atm", "col", "place", "trajet", "locp", "actp", "etatp", "senc", "obs", "obsm", "choc", "manv", "catr", "circ", "vosp",
"prof", "plan", "surf", "infra", "situ", "atm", "col", "dep", "com"]

for col in col_to_convert_object:
    df[col] = df[col].astype('str')


# On drop les variables non utilisées
df = df.drop(["num_veh", "Unnamed: 0", "num_acc", "date", "age_group"], axis = 1)


# ------------ Target Variable Regroupment ------------#

# On regroupe les modalités de notre variable target. Ce regroupement a été décidé après de multiples tests.

labels = {
    1: 0, # Indemne
    2: 2, # Deces
    3: 2, # Hospitalisé
    4: 1 # Léger
}

df['grav'] = df['grav'].map(labels)



                                #################################################
                                ############## Feature Selection ################
                                #################################################

# Cette étape regroupe les différentes méthodes afin de faire notre feature selection. Elle est composée de plusieurs test et méthodes : corrélation, selectKbest, selectFromModel.

# ------------ Corrélation Method ------------#

corr_matrix = df.corr()
plt.figure(figsize=(20, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

# On ne prend pas de décision à ce stade.

# ------------ Encoding ------------#

# On crée une fonction pour transformer nos modalités catégorielle en one hot encoding afin d'alimenter nos modèles.
def convert_category_to_dummies(X):
    cat_columns = X.select_dtypes(include = ["object"]).columns
    X_dummies = pd.get_dummies(X[cat_columns], drop_first = True, prefix = cat_columns)
    X_final = pd.concat([X.drop(columns = cat_columns), X_dummies], axis = 1)
    return X_final

df = convert_category_to_dummies(df)


# ------------ Train / Test Split ------------#

# On définit nos variables explicatives et la target de notre modèle.
X = df.drop("grav", axis = 1)
y = df.grav

# On sépare nos données en set d'entrainement, et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# ------------ SelectFromModel Method ------------#

# Fonction qui a pour but de tester différents seuils et modèles avec la méthode SelectFromModel afin de trouver le meilleur set de variable pour chaque modèle
def feature_selection_with_model(X_train, y_train, X_test, y_test, model_type, thresholds):

    scores = []
    selected_features = []

    # Sélection du modèle
    if model_type == "xgboost":
        model = xgb.XGBClassifier(random_state = 42)
    elif model_type == "lightgbm":
        model = lgb.LGBMClassifier(random_state = 42)
    elif model_type == "random_forest":
        model = RandomForestClassifier(random_state = 42, criterion = "gini")
    else:
        raise ValueError("Erreur, pas le bon model renseigné")

    # Entraînement du modèle
    model.fit(X_train, y_train)

    # Boucle pour tester différents seuils
    for threshold in thresholds:
        sfm = SelectFromModel(estimator=model, threshold=threshold)
        sfm.fit(X_train, y_train)
        

        X_train_transformed = sfm.transform(X_train)
        X_test_transformed = sfm.transform(X_test)
        clf = model.__class__(random_state = 42)
        clf.fit(X_train_transformed, y_train)
        y_pred = clf.predict(X_test_transformed)
        score = accuracy_score(y_test, y_pred)
        
        scores.append(score)
        selected_features.append(X_train.columns[sfm.get_support()].tolist())

        print(f'Threshold: {threshold}')
        print(f'Selected features: {selected_features[-1]}')
        print(f'Score: {score}\n')

    # On trave le graphique avec les différents test
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, scores, marker='o', linestyle='--')
    plt.title("Score en fonction du seuil de sélection des caractéristiques")
    plt.xlabel("Seuil de sélection des caractéristiques")
    plt.ylabel("Score de précision")
    plt.grid(True)
    plt.show()

    # On retourne les features du set de variable avec les meilleures perf
    best_index = np.argmax(scores)
    best_features = selected_features[best_index]
    print(f'Best threshold: {thresholds[best_index]}')
    print(f'Best features: {best_features}')
    
    return best_features

# ------------ SelectFromModel Application ------------#

# Random Forest (concluant)
model_type = "random_forest"
thresholds_rf = np.linspace(0.007, 0.013, num=5).tolist()
thresholds_rf.append(0.00925)
thresholds_rf = sorted(thresholds_rf)
best_features_rf = feature_selection_with_model(X_train, y_train, X_test, y_test, model_type, thresholds_rf)

# LGBM (non concluant)
'''
model_type = "lightgbm"
thresholds_lgb = np.linspace(1, 100, num=10).tolist()
thresholds_lgb = sorted(thresholds_lgb)
best_features_lgb = feature_selection_with_model(X_train, y_train, X_test, y_test, model_type, thresholds_lgb)
'''

# xgboost (non concluant)
'''
model_type = "xgboost"
thresholds_xgb = np.linspace(0.001, 0.003, num=5).tolist()
thresholds_xgb = sorted(thresholds_xgb)
best_features_xgb = feature_selection_with_model(X_train, y_train, X_test, y_test, model_type, thresholds_xgb)
'''


# ------------ SelectKBest Application ------------#

# Ici, on fait la feature selection avec du SelectKBest qui selectionne les K meilleures caractéristiques en se basant sur des test stats univariés, donc assez indépendant par rapport à la variable cible

# list of nb of features selection
k_values = np.linspace(10, 100, 10, dtype = int).tolist()
scores = []
selected_features = []

# on teste différents groupes de variables k , on veut chercher le seuil avec la meilleure accuracy (on utilise ici un rdf)
for k in k_values:
    skb = SelectKBest(score_func = f_classif, k = k)
    X_train_transformed = skb.fit_transform(X_train, y_train)
    X_test_transformed = skb.transform(X_test)
    clf = RandomForestClassifier(random_state = 42)
    clf.fit(X_train_transformed, y_train)
    y_pred = clf.predict(X_test_transformed)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)
    selected_features.append(X_train.columns[skb.get_support()].tolist())
    print(f'k: {k}')
    print(f'Selected features: {selected_features[-1]}')
    print(f'Score: {score}\n')

# on trace le graphique de l'accuracy en fonction de k
plt.figure(figsize=(10, 6))
plt.plot(k_values, scores, marker='o', linestyle='--')
plt.title("Score en fonction du nombre de caractéristiques sélectionnées (k)")
plt.xlabel("Nombre de caractéristiques sélectionnées (k)")
plt.ylabel("Score de précision")
plt.grid(True)
plt.show()

# on affiche les variables selectionnées pour chaque k. On retiendra les variables du modèle le plus performant.
for i, k in enumerate(k_values):
    print(f'k: {k}')
    print(f'Selected features: {selected_features[i]}')
    
    
# ------------ Final Feature Selection ------------#

# Après tous les test, on a décidé de garder ces sets de features à essayer.
# Features selection issues de la méthode SelectFromModel avec random forest

X_train_sfm_1 = X_train[['lartpc', 'larrout', 'mois', 'jour', 'hrmn', 'age', 'place_1.0', 'sexe_2', 'trajet_0.0', 'trajet_1.0', 'trajet_4.0', 'trajet_5.0', 'choc_1.0', 'manv_1.0', 'catr_3.0', 'agg_2']]
X_test_sfm_1 = X_test[['lartpc', 'larrout', 'mois', 'jour', 'hrmn', 'age', 'place_1.0', 'sexe_2', 'trajet_0.0', 'trajet_1.0', 'trajet_4.0', 'trajet_5.0', 'choc_1.0', 'manv_1.0', 'catr_3.0', 'agg_2']]

X_train_sfm_2 = X_train[['lartpc', 'larrout', 'mois', 'jour', 'hrmn', 'age', 'place_1.0', 'sexe_2', 'trajet_0.0', 'trajet_4.0', 'trajet_5.0', 'catr_3.0', 'agg_2']]
X_test_sfm_2 = X_test[['lartpc', 'larrout', 'mois', 'jour', 'hrmn', 'age', 'place_1.0', 'sexe_2', 'trajet_0.0', 'trajet_4.0', 'trajet_5.0', 'catr_3.0', 'agg_2']]

# Ces variables proviennent d'une feature selection que nous avions faites lorsque nous avions regroupé les moodalités de la variable target différemment. Méthode utilisée Selectkbest
X_train_k60 = X_train[['age', 'place_1.0', 'place_2.0', 'catu_2', 'catu_3', 'sexe_2', 'trajet_2.0', 'trajet_4.0', 'trajet_5.0', 'locp_0.0', 'locp_1.0', 'locp_2.0', 'locp_3.0', 'locp_4.0', 'actp_0.0', 'actp_1.0', 'actp_3.0', 'etatp_0.0', 'etatp_1.0', 'etatp_2.0', 'senc_1.0', 'catv_33', 'obs_0.0', 'obs_13.0', 'obs_2.0', 'obs_6.0', 'obs_8.0', 'obsm_0.0', 'obsm_2.0', 'choc_4.0', 'manv_13.0', 'manv_2.0', 'manv_23.0', 'catr_3.0', 'catr_4.0', 'circ_1.0', 'circ_2.0', 'prof_0.0', 'prof_1.0', 'plan_1.0', 'plan_2.0', 'plan_3.0', 'situ_1.0', 'situ_3.0', 'lum_3', 'agg_2', 'int_1', 'int_3', 'col_2.0', 'col_3.0', 'col_4.0', 'col_5.0', 'col_6.0', 'col_7.0', 'com_55', 'dep_130', 'dep_750', 'dep_910', 'dep_920', 'dep_940']]
X_test_k60 = X_test[['age', 'place_1.0', 'place_2.0', 'catu_2', 'catu_3', 'sexe_2', 'trajet_2.0', 'trajet_4.0', 'trajet_5.0', 'locp_0.0', 'locp_1.0', 'locp_2.0', 'locp_3.0', 'locp_4.0', 'actp_0.0', 'actp_1.0', 'actp_3.0', 'etatp_0.0', 'etatp_1.0', 'etatp_2.0', 'senc_1.0', 'catv_33', 'obs_0.0', 'obs_13.0', 'obs_2.0', 'obs_6.0', 'obs_8.0', 'obsm_0.0', 'obsm_2.0', 'choc_4.0', 'manv_13.0', 'manv_2.0', 'manv_23.0', 'catr_3.0', 'catr_4.0', 'circ_1.0', 'circ_2.0', 'prof_0.0', 'prof_1.0', 'plan_1.0', 'plan_2.0', 'plan_3.0', 'situ_1.0', 'situ_3.0', 'lum_3', 'agg_2', 'int_1', 'int_3', 'col_2.0', 'col_3.0', 'col_4.0', 'col_5.0', 'col_6.0', 'col_7.0', 'com_55', 'dep_130', 'dep_750', 'dep_910', 'dep_920', 'dep_940']]




                                #################################################
                                ############## Feature Selection ################
                                #################################################


            
# ------------ Modelization Function ------------#

# Fonction qui va nous permettre de tester plusieurs modèles "bruts" sans chercher les meilleurs hyperparamètres de chaque modèle.
def train_and_evaluate_model(model_name, X_train, X_test, y_train, y_test):
    # Ici, on a décidé d'implémenter plusieurs modèles "non hyperparamètrés" afin d'avoir une idée des performances de chaque modèle.
    if model_name == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == 'xgboost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', objective='multi:softprob',  num_class=3, max_delta_step=1,)
    elif model_name == 'lightgbm':
        model = lgb.LGBMClassifier()
    else:
        raise ValueError("Error: unsupported model")

    # On entraine le modèle et on prédit les valeurs
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # On calcule un set de métriques, pour la classif, le plus important pour nous sont accuracy, recall, f1 score.
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_df = pd.DataFrame(conf_matrix,
                                  index=['Observation Indemne', "Observation Leger", 'Observation Grave'],
                                  columns=['Prediction Indemne', "Prediction Leger", 'Prediction Grave'])

    # On affiche toutes les métriques
    print(f'Model: {model_name}')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Recall (global): {recall:.2f}')
    print(f'F1-Score (global): {f1:.2f}')
    print('Classification Report:')
    print(report)
    print('Confusion Matrix:')
    display(conf_matrix_df)

    conf_matrix_percent = conf_matrix_df / conf_matrix_df.sum().sum() * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_percent, annot=True, fmt=".2f", cmap='Blues', cbar=True)
    plt.title("Matrice de Confusion de l'accuracy en Pourcentages")
    plt.xlabel('Prédictions')
    plt.ylabel('Observation')
    plt.show()
    
    return model

# Cette fonction recherche le modèle random forest avec les meilleurs hyperparamètres à partir d'un gridsearch. On a restreint les paramètres car trop lourd sinon
def optimize_and_evaluate_random_forest(X_train, X_test, y_train, y_test):
    param_dist = {
        "n_estimators": [300, 400],
        "max_features": ["auto", "sqrt", "log2"],
        "max_depth": [None, 10, 20, 30, 40, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False]
    }

    rf = RandomForestClassifier(random_state = 42)

    random_search = RandomizedSearchCV(estimator = rf, param_distributions = param_dist,
                                       n_iter = 100, cv = 3, verbose = 2, random_state = 42, n_jobs = -1)

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    print("Best parameters found: ", best_params)
    best_rf = random_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_df = pd.DataFrame(conf_matrix,
                                  index=['Observation Indemne', "Observation Leger", 'Observation Grave'],
                                  columns=['Prediction Indemne', "Prediction Leger", 'Prediction Grave'])

    return accuracy, report, conf_matrix_df, best_params, best_rf


# ------------ Modelization Application ------------#

'''

# LightGBM : non concluant
model_name = 'lightgbm'
model_lgbm_basique = train_and_evaluate_model(model_name, X_train, X_test, y_train, y_test)
#model_lgbm_basique = joblib.load('../models/model_lgbm_basique.joblib')

# Xgboost : non concluant
model_name = 'xgboost'
model_xgb = train_and_evaluate_model(model_name, X_train, X_test, y_train, y_test)
#model_xgb_basique = joblib.load('../models/model_xgb_basique.joblib')

# Random Forest : concluant

# Etape 1 : modèle brut avec toutes les features
model_name = 'random_forest'
model_rf_basique = train_and_evaluate_model(model_name, X_train, X_test, y_train, y_test)
#model_rf_basique = joblib.load('../models/model_rf_basique.joblib')

# Etape 2 : modèle brut avec feature selection de SelectFromModel (seuil = 0.01)
model_name = 'random_forest'
model_rf_sfm_1 = train_and_evaluate_model(model_name, X_train_sfm_1, X_test_sfm_1, y_train, y_test)
#model_rf_sfm_1 = joblib.load('../models/model_rf_sfm_1.joblib')

# Etape 3 : modèle brut avec feature selection de SelectFromModel (seuil = 0.00925)
model_name = 'random_forest'
model_rf_sfm_2 = train_and_evaluate_model(model_name, X_train_sfm_2, X_test_sfm_2, y_train, y_test)
#model_rf_sfm_2 = joblib.load('../models/model_rf_sfm_2.joblib')
'''
# Etape 3 : modèle avec hyperparamètres (gridsearch) avec feature selection de SelectFromModel (seuil = 0.00925)
accuracy, report, conf_matrix_df, best_params, model_rf_sfm_2_best = optimize_and_evaluate_random_forest(X_train_sfm_2, X_test_sfm_2, y_train, y_test)
#model_rf_sfm_2_best = joblib.load('models/model_rf_sfm_2_best.joblib')



                                #################################################
                                ################ Interpretation #################
                                #################################################

        
# On définit les paramètres pour cette section (avec les différents test, il était plus facile de définir en amont)
model_lime = model_rf_sfm_2_best
X_train_lime = X_train_sfm_2
X_test_lime = X_test_sfm_2

# ------------ Feature Importance ------------#

importances = model_lime.feature_importances_
indices = np.argsort(importances)[::-1]

feature_names = X_train_lime.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# on affiche les features les plus importantes de notre modèle avec le score d'importance
print(feature_importance_df)

# on plot le résultat (pour l'oral)
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train_lime.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train_lime.shape[1]), feature_names[indices], rotation=90)
plt.tight_layout()
plt.show()




# ------------ LIME Interpretation of predicted values ------------#

# L'idée dans cette partie est de réussir à expliquer pour quelle raison notre modèle prédit pour telle observation, cette classe de risque.
# A noter que nous n'avons pas été en mesure d'implémenter les SHAP values. De ce fait, LIME nous donne uniquement une vision assez univarié de l'effet des variables sur la target. Une notion multivariée aurait été interessante.

def expliquer_instance(model, explainer, data, index, class_names):

    instance = data.iloc[index].values

    predicted_class = model.predict([instance])[0]
    predicted_class_name = class_names[predicted_class]
    
    print(f"Classe prédite pour l'instance {index} : {predicted_class_name} (Classe {predicted_class})")
    
    predicted_probabilities = model.predict_proba([instance])[0]
    for class_name, probability in zip(class_names, predicted_probabilities):
        print(f"Probabilité de la classe '{class_name}': {probability:.2f}")
    
    # partie explication de l'instance
    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba,
        num_features=13
    )
    
    # partie graphique
    plt.figure()
    exp.as_pyplot_figure()
    plt.title(f"Explication de la prédiction pour l'instance {index} : {predicted_class_name}")
    plt.show()

    
# ------------ LIME Application ------------#

# Initialisation de LimeTabularExplainer
explainer = LimeTabularExplainer(
    training_data=np.array(X_train_lime),
    feature_names=X_train_lime.columns,
    class_names=['Indemne', 'Léger', 'Grave'],
    mode='classification'
)

# Explication pour prédiction "Indemne"
expliquer_instance(model_lime, explainer, X_train_lime, index=2, class_names=['Indemne', 'Léger', 'Grave'])

# Explication pour prédiction "Léger"
expliquer_instance(model_lime, explainer, X_train_lime, index=6, class_names=['Indemne', 'Léger', 'Grave'])

# Explication pour prédiction "Grave"
expliquer_instance(model_lime, explainer, X_train_lime, index=8, class_names=['Indemne', 'Léger', 'Grave'])
