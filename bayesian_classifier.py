# """
# CON QUESTO COMBINAZIONE ABBIAMO OTTENUTO UN'ACCURATEZZA MEDIA DI 0.5463963963963965, ANCHE DOPO IL TUNING
# """

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold, GridSearchCV
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.naive_bayes import ComplementNB
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from data_standardization import standardization

# # Carico il dataset
# data = pd.read_csv('world_population.csv')

# # Rimuovo colonne non necessarie
# data = data.drop(['CCA3', 'Country/Territory', 'Capital', 'Rank'], axis=1)

# # --- FEATURE ENGINEERING ---
# population_cols = ['2022 Population', '2020 Population', '2015 Population', '2010 Population',
#                    '2000 Population', '1990 Population', '1980 Population', '1970 Population']

# # Media delle popolazioni (teniamola)
# data['Pop_Mean'] = data[population_cols].mean(axis=1)

# # Nuove features: combinazioni
# data['Growth_Rate_x_Density'] = data['Growth Rate'] * data['Density (per km²)']
# data['Pop_2022_over_Mean'] = data['2022 Population'] / data['Pop_Mean']
# data['Pop_2022_over_Area'] = data['2022 Population'] / data['Area (km²)']  # Densità 2022
# data['Pop_2022_over_2010'] = data['2022 Population'] / data['2010 Population']  # Rapporto 2022/2010
# data['Pop_2010_over_2000'] = data['2010 Population'] / data['2000 Population'] #Rapporto 2010/2000

# data = data[data['Area (km²)'] > 1000]

# X = data.drop('Continent', axis=1)
# y = data['Continent']

# # --- MODELLO (ComplementNB, SENZA SMOTE, SENZA SCALING per ora) ---

# # Inizializza il modello ComplementNB
# model = ComplementNB()

# # --- CROSS-VALIDATION ---
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# scores = []
# for train_index, test_index in cv.split(X, y):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#     print("\n--- Fold ---")
#     print("X_train shape:", X_train.shape)
#     print("X_test shape:", X_test.shape)


#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     accuracy = accuracy_score(y_test, y_pred)
#     scores.append(accuracy)

#     print(f"Fold Accuracy: {accuracy}")
#     print(classification_report(y_test, y_pred, zero_division=0))
#     print(confusion_matrix(y_test, y_pred))

# print(f"\nMedia Accuratezza (Cross-Validation): {np.mean(scores)}")
# # --- GRID SEARCH (dopo aver fatto feature engineering e aver deciso se usare SMOTE) ---

# param_grid = {
# 'alpha': [0.01, 0.1, 1.0, 10.0],
# 'norm': [True, False]
# }

# grid_search = GridSearchCV(model, param_grid, cv=cv, scoring="accuracy", verbose=1, n_jobs=-1)
# grid_search.fit(X, y)

# print(f"Migliori parametri: {grid_search.best_params_}")
# print(f"Migliore accuratezza: {grid_search.best_score_}")
# best_model = grid_search.best_estimator_
# # --- VALUTAZIONE FINALE (dopo aver fatto feature engineering e aver deciso se usare SMOTE) ---
# y_pred_final = best_model.predict(X)
# print(classification_report(y, y_pred_final, zero_division=0))
# print(confusion_matrix(y, y_pred_final))



import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from data_standardization import standardization

# Carico il dataset
data = pd.read_csv('world_population.csv')

# --- FEATURE ENGINEERING ---
population_cols = ['2022 Population', '2020 Population', '2015 Population', '2010 Population',
                   '2000 Population', '1990 Population', '1980 Population', '1970 Population']
data['Pop_Mean'] = data[population_cols].mean(axis=1)
data['Growth_Rate_x_Density'] = data['Growth Rate'] * data['Density (per km²)']
data['Pop_2022_over_Mean'] = data['2022 Population'] / data['Pop_Mean']
data['Pop_2022_over_Area'] = data['2022 Population'] / data['Area (km²)']
data['Pop_2022_over_2010'] = data['2022 Population'] / data['2010 Population']
data['Pop_2010_over_2000'] = data['2010 Population'] / data['2000 Population']

# --- PREPROCESSING ---
# Rimuovo l'outlier PRIMA del preprocessing
data = data[data['Area (km²)'] > 1000] 

X_processed, y, preprocessor = standardization(
    data,
    target_column='Continent',
    scaler_type='minmax',
    handle_missing='drop',
    categorical_encoding = 'onehot'
)

# --- MODELLO (ImbPipeline:  SMOTE + ComplementNB) ---
pipeline = ImbPipeline(steps=[
    ('smote', SMOTE(random_state=42)),
    ('classifier', ComplementNB())
])

# --- CROSS-VALIDATION ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- GRID SEARCH ---
param_grid = {
    'classifier__alpha': [0.01, 0.1, 1.0, 10.0],
    'classifier__norm': [True, False]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_processed, y)  # Usa X_processed e y

print(f"Migliori parametri: {grid_search.best_params_}")
print(f"Migliore accuratezza (CV): {grid_search.best_score_}")

# --- Valutazione e analisi degli errori all'interno della Cross Validation ---
scores = []
all_y_true = []
all_y_pred = []

for train_index, test_index in cv.split(X_processed, y): #Uso X_processed
    X_train, X_test = X_processed[train_index], X_processed[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Addestra SOLO il classificatore, NON l'intera pipeline (perché SMOTE è già nella pipeline)
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)  # Addestra il miglior modello (con i migliori iperparametri)
    y_pred = best_model.predict(X_test) #Predizioni sul test set

    accuracy = accuracy_score(y_test, y_pred)
    scores.append(accuracy)
    
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    print(f"Fold Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(confusion_matrix(y_test, y_pred))


print(f"\nMedia Accuratezza (Cross-Validation): {np.mean(scores)}")
print("\nAnalisi degli Errori (Cross-Validation):")
print(classification_report(all_y_true, all_y_pred, zero_division=0)) #Utilizza dati aggregati
print(confusion_matrix(all_y_true, all_y_pred)) #Utilizza dati aggregati
