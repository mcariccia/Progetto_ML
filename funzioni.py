import tkinter as tk
from tkinter import messagebox, ttk, scrolledtext
import pandas as pd
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def TuningIperparametri(classificatore, param_grid, x_train, y_train):
    grid_search = GridSearchCV(classificatore, param_grid, cv=5, refit=True, verbose=2)
    grid_search.fit(x_train, y_train)
    print("Migliori parametri trovati:", grid_search.best_params_)
    return grid_search.best_estimator_

def TuningConCrossValidation(classificatore, param_grid, x_train, y_train):
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=5, refit=True, verbose=2)
    grid_search.fit(x_train, y_train)

    # Migliori parametri dal grid search
    print(f"Migliori parametri: {grid_search.best_params_}")

    # Esegui la cross-validation con i migliori iperparametri
    cv_scores_with_tuning = grid_search.cv_results_['mean_test_score'][grid_search.best_index_]
    return cv_scores_with_tuning



def CrossValidation(classificatore, x_train, y_train):
    cv_scores = cross_val_score(classificatore, x_train, y_train, cv=5) 
    return cv_scores
