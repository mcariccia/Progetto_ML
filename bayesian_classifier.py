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
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from data_standardization import standardization

def naiveBayes():
    # Carico il dataset
    data = pd.read_csv('world_population.csv')

    data = data.drop(['CCA3', 'Country/Territory', 'Capital', 'Rank'], axis=1)

    # --- FEATURE ENGINEERING ---
    population_cols = ['2022 Population', '2020 Population', '2015 Population', '2010 Population',
                    '2000 Population', '1990 Population', '1980 Population', '1970 Population']
    data['Pop_Mean'] = data[population_cols].mean(axis=1)
    data['Growth_Rate_x_Density'] = data['Growth Rate'] * data['Density (per km²)']
    data['Pop_2022_over_Mean'] = data['2022 Population'] / data['Pop_Mean']
    data['Pop_2022_over_Area'] = data['2022 Population'] / data['Area (km²)']
    data['Pop_2022_over_2010'] = data['2022 Population'] / data['2010 Population']
    data['Pop_2010_over_2000'] = data['2010 Population'] / data['2000 Population']

    #--- PREPROCESSING ---
    #Rimuovo l'outlier PRIMA del preprocessing
    data = data[data['Area (km²)'] > 1000] 

    X_processed, y, preprocessor = standardization(
        data,
        target_column='Continent',
        scaler_type='minmax',
        handle_missing='drop',
        categorical_encoding = 'onehot'
    )

    # --- DIVISIONE TRAIN/TEST INIZIALE ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Dimensioni set di training: {X_train.shape}")
    print(f"Dimensioni set di test: {X_test.shape}")

    # --- MODELLO (ImbPipeline:  SMOTE + ComplementNB) ---
    pipeline = ImbPipeline(steps=[
        ('smote', SMOTE(random_state=42, k_neighbors=2)),
        ('classifier', ComplementNB())
    ])

    # --- GRID SEARCH CON CROSS-VALIDATION (sul set di training) ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        'classifier__alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
        'classifier__norm': [True, False]
    }

    print("Avvio grid search con 5-fold cross-validation...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1, return_train_score=True)
    grid_search.fit(X_train, y_train)  # Grid search solo sui dati di training 

    # --- RISULTATI TUNING ---
    print(f"\nMigliori parametri trovati: {grid_search.best_params_}")
    print(f"Migliore accuratezza (CV): {grid_search.best_score_:.4f}")

    # --- VALUTAZIONE SUL TEST SET ---
    best_model = grid_search.best_estimator_  # Questo modello è già addestrato sul set di training completo
    y_pred = best_model.predict(X_test)

    # --- METRICHE DI VALUTAZIONE FINALE ---
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n=== VALUTAZIONE FINALE ===")
    print(f"Accuratezza sul test set: {test_accuracy:.4f}")
    print("\nReport di classificazione:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\nMatrice di confusione:")
    print(confusion_matrix(y_test, y_pred))


def raw_naiveBayes():
    # Carico il dataset
    data = pd.read_csv('world_population.csv')

    # Rimozione di colonne non necessarie
    data = data.drop(['CCA3', 'Country/Territory', 'Capital', 'Rank'], axis=1)

    # Rimozione di outlier
    data = data[data['Area (km²)'] > 1000]

    # Divisione in feature e target
    X = data.drop('Continent', axis=1)
    y = data['Continent']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Addestramento del modello base
    model = ComplementNB()
    model.fit(X_train, y_train)

    # Valutazione
    y_pred = model.predict(X_test)
    base_accuracy = accuracy_score(y_test, y_pred)

    # Cross-validation per una stima più robusta
    cv_scores = cross_val_score(model, X, y, cv=5)

    print(f"Accuratezza sul test set: {base_accuracy:.4f}")
    print(f"Accuratezza media CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\nReport di classificazione:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
raw_naiveBayes()