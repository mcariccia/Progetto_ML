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

        if cross_validation:
            cv_scores = CrossValidation(best_clf, x_train, y_train)
            print(f"Score della cross-validation: {cv_scores}")
            print("Cross-validation completata")
            accuracy = cv_scores.mean()
        else:
            best_clf.fit(x_train, y_train)
            y_predv = best_clf.predict(x_test)
            accuracy = accuracy_score(y_test, y_predv)

        print(f"Accuratezza del modello: {accuracy:.2f}")
        print("Modello addestrato")

    elif classificatore == "ensemble":
        print()
    elif classificatore == "kmeans":
        print()
    elif classificatore == "knn Custom":
        def calcolo_distanze(x_train, x_test, distanza):
            distanze = []
            for x_t in x_test:
                if distanza == "euclidea":
                    dist = np.sqrt(np.sum(np.square(x_train - x_t), axis=1))
                elif distanza == "manhattan":
                    dist = np.sum(np.abs(x_train - x_t), axis=1)
                elif distanza == "chebyshev":
                    dist = np.max(np.abs(x_train - x_t), axis=1)
                distanze.append(dist)
            return np.array(distanze)
        

        def knn(y_train, k, valdist, param_grid):
            k_min = np.argsort(valdist)[:k]
            if param_grid == "uniform":
                y_pred = np.argmax(np.bincount(y_train[k_min]))
            elif param_grid == "distance":
                weights = 1 / (valdist[k_min] + 1e-5)  # Aggiungi un piccolo valore per evitare divisioni per zero
                y_pred = np.argmax(np.bincount(y_train[k_min], weights=weights))
            return y_pred
        
        
        param_grid = {
            'pesi': ['uniform', 'distance']
        }   
        
        if tuning:
            best_clf = TuningIperparametri(knn(y_train, x_test, calcolo_distanze(x_train, x_test, distanza), param_grid), param_grid, x_train, y_train)
        else:
            best_clf = knn(y_train, valk, calcolo_distanze(x_train, x_test, distanza), param_grid)

        if cross_validation:
            cv_scores = cross_val_score(best_clf, x_train, y_train, cv=5)
            accuracy = cv_scores.mean()
        else:
            best_clf.fit(x_train, y_train)
            y_predv = best_clf.predict(x_test)
            
            
        




    return accuracy
