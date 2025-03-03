import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def standardization(data, target_column, scaler_type='minmax', handle_missing='drop', categorical_encoding='onehot'):
    """
    Funzione per la standardizzazione dei dati.
    
    Args: 
    - data (pd.DataFrame): pandas DataFrame contenente i dati
    - target_column (str): stringa contenente il nome della colonna target
    - scaler_type (str, optional): stringa contenente il tipo di scaler da utilizzare (standard, minmax, robust o None. Default: minmax)
    - handle_missing (str, optional): come gestire i valori mancanti. Può essere: 
        - 'drop': rimuove le righe con valori mancanti
        - 'mean': sostituisce i valori mancanti con la media della colonna
        - 'median': sostituisce i valori mancanti con la mediana della colonna
        - 'most_frequent': sostituisce i valori mancanti con il valore più frequente della colonna
        - 'constant': sostituisce i valori mancanti con un valore costante (specificato nel parametro fill_value)
        - None: non fa nulla
        - Default: 'drop'
    - categorical_encoding (str, optional): come gestire le variabili categoriche. Può essere:
        - 'onehot': codifica le variabili categoriche con OneHotEncoder
        - None: non fa nulla (le colonne categoriche vengono ignorate)
        - Default: 'onehot'
    
    Returns:
    - Una tupla contenente: 
        - X_processed (np.array o pd.DataFrame): array o DataFrame contenente i dati processati
        - y (pd.Series): serie contenente i valori target
        - preprocessor (ColumnTransformer): oggetto ColumnTransformer contenente le trasformazioni applicate
    """
    
    X = data.drop(columns=target_column)
    y = data[target_column]
    
    # Identifica le colonne numeriche e categoriche
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
    
    # Gestione dei valori mancanti, prima dello Scaling
    if handle_missing == 'drop':
        X = X.dropna()
        y = y.loc[X.index]  # Importante per mantenere l'allineamento tra X e y
    elif handle_missing in ['mean', 'median', 'most_frequent']:
        for col in numerical_features: 
            if X[col].isnull().any():   #controlla se ci sono NaN in questa colonna
                if handle_missing == 'mean':
                    fill_value = X[col].mean()
                elif handle_missing == 'median':
                    fill_value = X[col].median()
                else:   # handle_missing == 'most_frequent'
                    fill_value = X[col].mode()[0]
                X[col] = X[col].fillna(fill_value)
    elif handle_missing == 'constant':
        fill_value = 0
        for col in numerical_features:
            if X[col].isnull().any():
                X[col] = X[col].fillna(fill_value)
                
    
    # Scaling e Encoding
    transformers = []
    if scaler_type is not None and len(numerical_features) > 0:
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else: 
            raise ValueError("Scaler_type invalido. Deve essere 'standard', 'minmax', 'robust' o None.")
        transformers.append(('num', scaler, numerical_features))
    
    if categorical_encoding == 'onehot' and len(categorical_features) > 0:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features))
    
    # Creazione del ColumnTransformer
    if len(transformers) > 0:
        preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
        X_processed = preprocessor.fit_transform(X)
        
        #Gestione output: lo vogliamo come DataFrame
        if isinstance(X_processed, np.ndarray):
            #Costruisco i nomi delle colonne
            columns_names = []
            for transformer_name, transformer, cols in preprocessor.transformers_:
                if transformer_name == 'num':
                    columns_names.extend(cols)
                elif transformer_name == 'cat':
                    columns_names.extend(transformer.get_feature_names_out(cols))
                else: #remainder
                    columns_names.extend(cols)
            X_processed = pd.DataFrame(X_processed, columns = columns_names)
      
    else:
        preprocessor = None
        X_processed = X
    
    return X_processed, y, preprocessor