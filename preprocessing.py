from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy import stats
import pandas as pd

def preprocessing(df, bilanciamento, standardizzazione, normalizzazione, selezione_features, rimozione_outlier):
    # Identifica le colonne numeriche e categoriali
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    x = df.drop(columns=['Continent'])
    y = df['Continent']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    if selezione_features:
        # Seleziona le migliori K caratteristiche solo sulle colonne numeriche
        select = SelectKBest(score_func=chi2, k=5)
        x_train_numeric = select.fit_transform(x_train[numeric_columns], y_train)
        x_test_numeric = select.transform(x_test[numeric_columns])
        selected_features = select.get_support(indices=True)
        selected_numeric_columns = [numeric_columns[i] for i in selected_features]
        print("Caratteristiche selezionate")
    else:
        x_train_numeric = x_train[numeric_columns]
        x_test_numeric = x_test[numeric_columns]
        selected_numeric_columns = numeric_columns

    if rimozione_outlier:
        # Rimuove gli outlier solo sulle colonne numeriche
        z = np.abs(stats.zscore(x_train_numeric))
        x_train_numeric = x_train_numeric[(z < 3).all(axis=1)]
        y_train = y_train.iloc[(z < 3).all(axis=1)]
        print("Outlier rimossi")

    if standardizzazione:
        # Standardizza le caratteristiche solo sulle colonne numeriche
        scaler = StandardScaler()
        x_train_numeric = scaler.fit_transform(x_train_numeric)
        x_test_numeric = scaler.transform(x_test_numeric)
        print("Caratteristiche standardizzate")

    if normalizzazione:
        # Normalizza le caratteristiche solo sulle colonne numeriche
        scaler = MinMaxScaler()
        x_train_numeric = scaler.fit_transform(x_train_numeric)
        x_test_numeric = scaler.transform(x_test_numeric)
        print("Caratteristiche normalizzate")

    if bilanciamento == "SMOTE":
        # Bilancia le classi nel set di addestramento utilizzando SMOTE
        smote = SMOTE(random_state=42)
        x_train_numeric, y_train = smote.fit_resample(x_train_numeric, y_train)
        print("Distribuzione delle classi nel set di addestramento bilanciato")

    elif bilanciamento == "Random Over Sampling":
        # Bilancia le classi nel set di addestramento utilizzando Random Over Sampling
        ros = RandomOverSampler(random_state=42)
        x_train_numeric, y_train = ros.fit_resample(x_train_numeric, y_train)
        print("Distribuzione delle classi nel set di addestramento bilanciato")

    elif bilanciamento == "Random Under Sampling":
        # Bilancia le classi nel set di addestramento utilizzando Random Under Sampling
        rus = RandomUnderSampler(random_state=42)
        x_train_numeric, y_train = rus.fit_resample(x_train_numeric, y_train)
        print("Distribuzione delle classi nel set di addestramento bilanciato")

    # Ricombina le colonne numeriche selezionate e categoriali
    x_train = pd.concat([pd.DataFrame(x_train_numeric, columns=selected_numeric_columns), x_train[categorical_columns].reset_index(drop=True)], axis=1)
    x_test = pd.concat([pd.DataFrame(x_test_numeric, columns=selected_numeric_columns), x_test[categorical_columns].reset_index(drop=True)], axis=1)

    return x_train, x_test, y_train, y_test