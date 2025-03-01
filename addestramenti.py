import tkinter as tk
from tkinter import messagebox, ttk, scrolledtext
import pandas as pd
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from funzioni import TuningIperparametri, CrossValidation

def AddestramentoClassificatori(df, classificatore, cross_validation, tuning, distanza, valk):
    # Seleziona le caratteristiche rilevanti e la variabile target
    x = df[['2022 Population', 'Area (km²)', 'Density (per km²)', 'Growth Rate']]
    y = df['Continent']

    # Dividi il dataset in set di addestramento e test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print("Dataset diviso in set di addestramento e test")

    # Bilancia le classi nel set di addestramento utilizzando SMOTE
    smote = SMOTE(random_state=42)
    x_train, y_train = smote.fit_resample(x_train, y_train)
    print("Distribuzione delle classi nel set di addestramento bilanciato:")

    # Standardizza le caratteristiche
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    print("Caratteristiche standardizzate")

    if classificatore == "svm":
        if tuning:
            best_clf = TuningIperparametri(svm.SVC(), x_train, y_train)
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

        return y_predv, accuracy
