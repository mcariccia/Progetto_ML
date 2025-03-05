
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from dataAnalysis import load_data, feat_label

# Caricare e Pre-elaborare il Dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)

    # Selezione delle feature numeriche
    feature_columns = ["1970 Population", "1980 Population", "1990 Population",
                       "2000 Population", "2010 Population", "2015 Population",
                       "2020 Population", "2022 Population", "Density (per km²)", "Growth Rate"]
    
    target_column = "Continent"

    #  Estrarre feature e target
    X = df[feature_columns].fillna(0)
    y = df[target_column]

    # Normalizzazione delle feature
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #  Conversione della variabile target in numeri
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print(f" Dataset caricato! Dimensioni: {df.shape}")
    print(f" Distribuzione delle classi target: {Counter(y_encoded)}")

    return X_scaled, y_encoded, encoder

def tuning():
    flag = 0
    dataset = load_data()
    X, y= feat_label(load_data())
    if flag == 0:
            X = X.drop(labels=['CCA3', 'Capital', 'Country/Territory'], axis=1)

    #  Dividere il dataset in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Eseguire il tuning degli iperparametri con cross-validation
    max_depth_range = list(range(1, 25))
    max_leaf_range = list(range(2,25))

    best_score = 0
    best_params = {}
    #  Scorrere i valori di max_depth e max_leaf_nodes
    for depth in max_depth_range:
        for leaf_nodes in max_leaf_range:
            clf = DecisionTreeClassifier(max_depth=depth, max_leaf_nodes=leaf_nodes, random_state=42)
            
            #  Eseguo la cross-validation con 10 folds
            scores = cross_validate(clf, X_train, y_train, cv=10, scoring='accuracy', return_train_score=True)

            #  Calcolo le medie delle accuracy per training e validation
            score_train = scores['train_score'].mean()
            score_val = scores['test_score'].mean()
            
            #  Aggiorno i migliori iperparametri se trovo un modello migliore
            if score_val > best_score:
                best_score = score_val
                best_params = {'max_depth': depth, 'max_leaf_nodes': leaf_nodes}

    #  Stampa dei migliori iperparametri trovati
    print(f"\n Migliori iperparametri trovati: {best_params}")
    print(f" Migliore accuracy media in cross-validation: {best_score:.4f}")
#Parametri migliori: max_depth=7, max_leaf=21

def decision_tree(X, y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    #Creazione del modello finale con gli iperparametri migliori
    best_tree = DecisionTreeClassifier(max_depth=7, max_leaf_nodes=21)
    best_tree.fit(X_train, y_train)

    # Eseguire la classificazione sul test set
    y_pred = best_tree.predict(X_test)

    #Calcolare l'accuratezza finale del modello
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n Accuratezza finale sul test set: {accuracy:.4f}")

    #  Stampa del Report di Classificazione
    print("\n Report di Classificazione:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))

    #  Matrice di confusione
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel("Predetto")
    plt.ylabel("Reale")
    plt.title("Matrice di Confusione")
    plt.show()

    # Visualizzazione dell'albero decisionale
    plt.figure(figsize=(20,10))
    plot_tree(best_tree, feature_names=["1970 Population", "1980 Population", "1990 Population",
                                        "2000 Population", "2010 Population", "2015 Population",
                                        "2020 Population", "2022 Population", "Rank", "Area", 
                                        "Density", "Growth Rate", "World Population Percentage"],
            class_names=encoder.classes_, filled=True, fontsize=8)
    plt.title("Albero Decisionale Ottimizzato")
    plt.show()
