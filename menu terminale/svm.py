import pandas as pd
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from dataAnalysis import load_data
from sklearn.svm import SVC

def tuning ():
    df = load_data()
    # Seleziona le caratteristiche rilevanti e la variabile target
    X = df[['2022 Population', 'Area', 'Density', 'Growth Rate']]
    y = df['Continent']

    # Dividi il dataset in set di addestramento e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definisci la griglia dei parametri
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }

    # Esegui il tuning degli iperparametri usando GridSearchCV con cross-validation
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=5, refit=True)
    grid_search.fit(X_train, y_train)

    # Migliori parametri dal grid search
    print(f"Migliori parametri: {grid_search.best_params_}")

    # Esegui la cross-validation con i migliori iperparametri
    cv_scores_with_tuning = grid_search.cv_results_['mean_test_score'][grid_search.best_index_]
    print(f"Cross-validation accuracy con tuning: {cv_scores_with_tuning}")

def svm_class (X, Y):
    X = X.drop(labels=['CCA3', 'Capital', 'Country/Territory'], axis=1)
    
    # Dividi il dataset in set di addestramento e test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Addestra il modello con i migliori parametri e fai previsioni
    best_clf = SVC(C=100, gamma=1, kernel='rbf')
    best_clf.fit(X_train, y_train)
    y_pred_tuned = best_clf.predict(X_test)
    
    # Calcola l'accuratezza con tuning
    acc_with_tuning = metrics.accuracy_score(y_test, y_pred_tuned)
    print(f"Accuratezza con tuning: {acc_with_tuning}")