import tkinter as tk
from tkinter import messagebox, ttk, scrolledtext
import pandas as pd
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from funzioni import TuningIperparametri, CrossValidation
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def AddestramentoClassificatori(x_train, x_test, y_train, y_test, classificatore, cross_validation, tuning, distanza, valk):

    # Bilancia le classi nel set di addestramento utilizzando SMOTE
    smote = SMOTE(random_state=42)
    x_train, y_train = smote.fit_resample(x_train, y_train)
    print("Distribuzione delle classi nel set di addestramento bilanciato:")

    # Standardizza le caratteristiche
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    print("Caratteristiche standardizzate")

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    if classificatore == "svm":
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
        }
        if tuning:
            best_clf = TuningIperparametri(svm.SVC(), param_grid, x_train, y_train)
            print("Tuning completato")        
            best_clf.fit(x_train, y_train)
            y_predv = best_clf.predict(x_test)
            accuracy = metrics.accuracy_score(y_test, y_predv)
        else:
            best_clf = svm.SVC()      
            best_clf.fit(x_train, y_train)
            y_predv = best_clf.predict(x_test)
            accuracy = metrics.accuracy_score(y_test, y_predv)

        if cross_validation:
            cv_scores = CrossValidation(best_clf, x_train, y_train)
            print(f"Score della cross-validation: {cv_scores}")
            print("Cross-validation completata")
            accuracy = cv_scores.mean()
            
        print("Modello addestrato")

    elif classificatore == "Naive Bayes":

        print("Valori minimi nel dataset di addestramento:", x_train.min(axis=0))
        print("Valori minimi nel dataset di test:", x_test.min(axis=0))
    

        param_grid = {
            'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
            'norm': [True, False]
        }

        if tuning:
            best_clf = TuningIperparametri(ComplementNB(), param_grid, x_train, y_train)
            print("Tuning completato")
        else:
            best_clf = ComplementNB()

        # Addestriamo il modello
        best_clf.fit(x_train, y_train)
        y_predv = best_clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_predv)

        if cross_validation:
            cv_scores = CrossValidation(best_clf, x_train, y_train)
            print(f"Score della cross-validation: {cv_scores}")
            print("Cross-validation completata")
            accuracy = cv_scores.mean()

        print(f"Accuratezza del modello: {accuracy:.2f}")
        print("Modello addestrato")

    elif classificatore == "ensemble":
        print()
    elif classificatore == "kmeans":
        print()
    elif classificatore == "knn Custom":
        print()

    return y_predv, accuracy
