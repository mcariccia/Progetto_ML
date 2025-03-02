import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, InstanceHardnessThreshold
from sklearn.model_selection import train_test_split

#  Funzione per il caricamento del dataset
def load_dataset(file_path):
     #caricao il dataset world_population.csv e seleziona feature e target.
    
     #Parametri:
    - file_path: #Percorso del file CSV.
    
     #Output:
    # X: Feature dataset (numpy array o Pandas DataFrame).
    # y: Target labels (numpy array o Pandas Series).
    
    #df = pd.read_csv(file_path)

    feature_columns = ["1970 Population", "1980 Population", "1990 Population",
                       "2000 Population", "2010 Population", "2015 Population",
                       "2020 Population", "2022 Population", "Density (per km²)", "Growth Rate"]

    target_column = "Continent"  # Puoi cambiare la colonna target in base alla tua analisi

    X = df[feature_columns].fillna(0)  # Sostituisce i valori NaN con 0
    y = df[target_column]

    print(f" Dataset caricato! Dimensioni: {df.shape}")
    print(f" Distribuzione delle classi target: {Counter(y)}")

    return X, y

#  Funzione per l’Undersampling
def apply_undersampling(X, y, method="random"):
    
    #Applica un metodo di Undersampling per ridurre la classe maggioritaria.

     #Parametri:
    # X: Feature dataset (numpy array o Pandas DataFrame).
    # y: Target labels (numpy array o Pandas Series).
    # method: Tipo di undersampling ('random', 'iht').

     #Output:
     # X_resampled: Feature bilanciate.
    # y_resampled: Etichette bilanciate.
    
    print(f"\n Dataset originale: {Counter(y)}")

    if method == "random":
        print("\n Applicando Random Undersampling...")
        sampler = RandomUnderSampler(random_state=42)
    elif method == "iht":
        print("\nApplicando Instance Hardness Threshold Undersampling...")
        sampler = InstanceHardnessThreshold()
    else:
        raise ValueError(" Metodo non supportato! Scegli tra 'random' o 'iht'.")

    X_resampled, y_resampled = sampler.fit_resample(X, y)
    print(f" Dataset dopo undersampling: {Counter(y_resampled)}")

    return X_resampled, y_resampled


#  Funzione per l’Oversampling
def apply_oversampling(X, y, method="random"):
    
     #aplica un metodo di Oversampling per aumentare la classe minoritaria.
    #parametri:
    #: Feature dataset (numpy array o Pandas DataFrame).
    #y: Target labels (numpy array o Pandas Series).
    # method: Tipo di oversampling ('random', 'smote').

    #Output:
    #X_resampled: Feature bilanciate.
    #y_resampled: Etichette bilanciate.
    
    print(f"\n Dataset originale: {Counter(y)}")

    if method == "random":
        print("\n Applicando Random Oversampling...")
        sampler = RandomOverSampler(random_state=42)
    elif method == "smote":
        print("\n Applicando SMOTE Oversampling...")
        sampler = SMOTE(random_state=42)
    else:
        raise ValueError(" Metodo non supportato! Scegli tra 'random' o 'smote'.")

    X_resampled, y_resampled = sampler.fit_resample(X, y)
    print(f" Dataset dopo oversampling: {Counter(y_resampled)}")

    return X_resampled, y_resampled


if __name__ == "__main__":
    #Carica il dataset
    file_path = "world_population.csv"  #percorso file dataset
    X, y = load_dataset(file_path)

    # Divido ataset in training e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # **Applica Undersampling**
    X_under, y_under = apply_undersampling(X_train, y_train, method="random")

    # **Applica Oversampling**
    X_over, y_over = apply_oversampling(X_train, y_train, method="smote")
