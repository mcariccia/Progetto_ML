import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder

# Funzione per caricare i dati dal file
def load_data():
    # Individua la posizione del file corrente - necessario per il percorso relativo
    path = Path(__file__).parent
    
    # Crea un dataframe con il dataset
    dataset = pd.read_csv((path/"world_population.csv").resolve())
    # Ridenominazione di attributi con nomi problematici
    dataset.rename(columns = {list(dataset)[-4]:'Area', list(dataset)[-3]:'Density'}, inplace=True)
    
    return dataset

# Funzione per estrarre attributi e classe
def feat_label (dataset):
    
    # Estrae gli attributi
    X_features = dataset.drop(labels=['Continent'], axis=1)
    # Estrae la classe
    Y_label = dataset['Continent']
    
    # Restituisce attributi e classe
    return X_features, Y_label

# Funzione per l'analisi del dataset
def analysis (X):
    # Estrazione dei nomi delle colonne
    feat_names = X.columns
    
    # Identificazione e gestione delle colonne categoriche:
    # Gli attributi categorici vengono convertiti in numerici
    X_numeric = X.copy() # Creazione di una copia del dataframe per non perdere l'originale
    encoder = LabelEncoder() # Creo l'encoder 
    for feat in feat_names: # Ciclo per scorrere la lista delle features
        if type(X_numeric[feat][0]) == type("H"): # Controllo se il primo elemento è di tipo stringa
            X_numeric[feat] = encoder.fit_transform(X_numeric[feat]) # Conversione 
    
    # Inizializzazione degli array
    means = []
    modes = []
    medians = []
    std = []
    var = []
    range = []
    tenth_perc = []
    twentyfifth_perc = []
    fiftieth_perc = []
    seventyfifth_perc = []
    ninetieth_perc = []
    
    # Ciclo per scorrere gli attributi - calcolando i valori per ognuno
    for feat in feat_names:
        means = means + [X_numeric[feat].mean()]
        mode_val, mode_count = mode(X_numeric[feat])
        modes = modes + [[mode_val.item(), mode_count.item()]]        
        medians = medians + [X_numeric[feat].median()]
        std = std + [X_numeric[feat].std()]
        var = var + [X_numeric[feat].var()]
        range = range + [X_numeric[feat].max() - X_numeric[feat].min()]
        tenth_perc = tenth_perc + [np.percentile(X_numeric[feat], 10)]
        twentyfifth_perc = twentyfifth_perc + [np.percentile(X_numeric[feat], 25)]
        fiftieth_perc = fiftieth_perc + [np.percentile(X_numeric[feat], 50)]
        seventyfifth_perc = seventyfifth_perc + [np.percentile(X_numeric[feat], 75)]
        ninetieth_perc = ninetieth_perc + [np.percentile(X_numeric[feat], 90)]
    
    # Calcolo della correlazione
    correlation = X_numeric.corr()
    
    # Restituzione delle liste dei valori ottenuti
    return means, modes, medians, std, var, range, tenth_perc, twentyfifth_perc, fiftieth_perc, seventyfifth_perc, ninetieth_perc, correlation

# Funzione per la stampa dei risultati ottenuti dalla precedente
def print_analysis(info):
    # Label per colonne e righe
    feat_names = ['Rank', 'CCA3', 'Country/Territory', 'Capital', '2022 Population', '2020 Population', '2015 Population', '2010 Population',
       '2000 Population', '1990 Population', '1980 Population', '1970 Population', 'Area', 'Density', 'Growth Rate',
       'World Population Percentage']
    
    operation = ['Mean', 'Mode', 'Median', 'Standard Deviation', 'Variance', 'Range', '10%', '25%', '50%', '75%', '90%']
    
    # Creazione del nuovo dataframe
    results = pd.DataFrame(info[:-1], columns=feat_names, index=operation)
    
    # Stampa dei risultati
    print("\nRisultati dell'analisi dei dati:")
    print(results)
    
    print("\nCorrelazione tra gli attributi:")
    print(info[-1])

# Funzione per creare i boxplot degli attributi numerici
def boxplot (X):
    # Ricerca degli attributi non numerici
    feat_names = X.columns
    
    cat = [] # Inizializzo lista vuota
    for feat in feat_names: # Ciclo per scorrere la lista delle features
        if type(X[feat][0]) == type("H"): # Controllo se il primo elemento è di tipo stringa
            cat = cat + [feat] # Se si, aggiunto alla lista dei categorici
    feat_names = [feat for feat in feat_names if feat not in cat] # Creazione di lista di attributi numerici
    
    # Creazione dei subplot
    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(30, 15))
    ax = ax.flatten() # Trasforma la matrice 3x5 in una lista
    
    # Titolo dei grafici
    fig.suptitle('Boxplot degli attributi - Senza outliers', fontsize=20)
    
    # Ciclo per creare i grafici degli attributi - senza outliers
    for i, feat in enumerate(feat_names):
        # Adatta gli assi ai dati
        plt.sca(ax[i])
        
        # Crea il plot senza outliers
        X[feat].plot(kind='box', patch_artist = True, showfliers=False, grid= True)

        # Individua i percentili 
        q25 = np.percentile(X[feat], 25)
        q50 = np.percentile(X[feat], 50)
        q75 = np.percentile(X[feat], 75)
        q90= np.percentile(X[feat], 90)
        perc_val = [q25, q50, q75, q90]

        # Disegna i percentili
        plt.scatter([1, 1, 1, 1], perc_val, marker='o', color='red', label='Percentiles')

        # Etichetta per i percentili
        perc_name = ["25%", "50%", "75%", "90%"]
        for i, val in enumerate(perc_val):
            plt.text(1.1, val, f"{perc_name[i]}: {val:.2f}")

        # Crea legenda per il grafico
        plt.legend(loc='upper left')
    
    # Cancella grafici vuoti
    fig.delaxes(ax[-2])
    fig.delaxes(ax[-1])
    
    # Mostra i risultati
    plt.show()
    
    # Creazione dei subplot
    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(30, 15))
    ax = ax.flatten() # Trasforma la matrice 3x5 in una lista
    
    # Titolo dei grafici
    fig.suptitle('Boxplot degli attributi - Con outliers', fontsize=20)
    
    # Grafici degli attributi con outliers
    for i, feat in enumerate(feat_names):
        # Adatta gli assi ai dati
        plt.sca(ax[i])
        
        # Crea il plot senza outliers
        X[feat].plot(kind='box', patch_artist = True, grid=True)
        
        # Individua i quantili calcolati nell'analisi
        q10 = np.percentile(X[feat], 10)
        q25 = np.percentile(X[feat], 25)
        q50 = np.percentile(X[feat], 50)
        q75 = np.percentile(X[feat], 75)
        q90= np.percentile(X[feat], 90)
        perc_val = [q10, q25, q50, q75, q90]

        # Disegna i quantili
        plt.scatter([1, 1, 1, 1, 1], perc_val, marker='o', color='red', label='Percentiles')

        # Etichetta per i quantili
        perc_name = ["10%", "25%", "50%", "75%", "90%"]
        for i, val in enumerate(perc_val):
            plt.text(1.1, val, f"{perc_name[i]}: {val:.2f}")

        # Crea legenda per il grafico
        plt.legend(loc='upper left')
    
    # Cancella grafici vuoti
    fig.delaxes(ax[-2])
    fig.delaxes(ax[-1])
    
    # Mostra i risultati
    plt.show()
  
# Funzione per creare istogrammi degli attributi numerici  
def histogram (X, Y):
    # Ricerca degli attributi non numerici
    feat_names = X.columns
    
    cat = [] # Inizializzo lista vuota
    for feat in feat_names: # Ciclo per scorrere la lista delle features
        if type(X[feat][0]) == type("H"): # Controllo se il primo elemento è di tipo stringa
            cat = cat + [feat] # Se si, aggiunto alla lista dei categorici
    feat_names = [feat for feat in feat_names if feat not in cat] # Creazione di lista di attributi numerici
    
    # Creazione dei subplot
    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(30, 30))
    ax = ax.flatten() # Trasforma la matrice 3x5 in una lista
    
    # Titolo dei grafici
    fig.suptitle('Istogramma degli attributi', fontsize=20)
    
    # Ciclo per creare i grafici degli attributi - senza outliers
    for i, feat in enumerate(feat_names):
        # Adatta gli assi ai dati
        plt.sca(ax[i])
        
        # Creazione del plot - suddiviso per continent
        X[feat][Y == 'Asia']         .plot(kind='hist', density= False, bins= 20, alpha= 0.5, label='Asia')
        X[feat][Y == 'Africa']       .plot(kind='hist', density= False, bins= 20, alpha= 0.5, label='Africa')
        X[feat][Y == 'Europe']       .plot(kind='hist', density= False, bins= 20, alpha= 0.5, label='Europe')
        X[feat][Y == 'Oceania']      .plot(kind='hist', density= False, bins= 20, alpha= 0.5, label='Oceania')
        X[feat][Y == 'North America'].plot(kind='hist', density= False, bins= 20, alpha= 0.5, label='North America')
        X[feat][Y == 'South America'].plot(kind='hist', density= False, bins= 20, alpha= 0.5, label='South America')

        # Crea legenda per il grafico
        plt.legend(loc='upper right')
        
        # Aggiunge il titolo all'istogramma
        plt.title(feat, fontsize= 10)
    
    # Cancella grafici vuoti
    fig.delaxes(ax[-2])
    fig.delaxes(ax[-1])
    
    # Mostra i risultati
    plt.show()

# Definizione della funzione per una stampa migliore delle percentuali nel grafico a torta
def pct_chart(pct, values):
    # Riceve la percentuale dalla funzione del grafico 
    # Riceve i dati di input del grafico
    # Usa entrambi per calcolare il numero di elementi per ogni "fetta"
    number = int(pct/100 * sum(values))
    # Restituisce la stringa che viene stampata su ciascuna fetta
    return f"{pct:.3f}%\n({number:d})"

# Funzione per un grafico a torta delle classi
def pie_chart (Y):
    # Calcola le occorrenze delle classi
    asia          = (Y[Y=='Asia'].count()         )
    africa        = (Y[Y=='Africa'].count()       )
    europe        = (Y[Y=='Europe'].count()       )
    oceania       = (Y[Y=='Oceania'].count()      )
    north_america = (Y[Y=='North America'].count())
    south_america = (Y[Y=='South America'].count())
    
    # Crea lista dei valori e dei label
    values = [asia, africa, europe, oceania, north_america, south_america]
    label = ['Asia', 'Africa', 'Europe', 'Oceania', 'North America', 'South America']
    
    # Crea il grafico    
    plt.pie(values, labels=label, autopct=lambda pct: pct_chart(pct, values))
    
    # Aggiunge il titolo
    plt.title("Proporzioni delle classi nel dataset", fontsize= 20)
    
    # Stampa il grafico
    plt.show()
   
# Funzione unica per eseguire tutte le analisi 
def data_analysis ():
    # Carica il dataset 
    dataset = load_data()
    # Divide il dataset in attributi e classi
    X, Y = feat_label(dataset)
    # Calcolo e stampa dei risultati dell'analisi matematica
    print_analysis(analysis(X))
    # Stampa dei grafici
    boxplot(X)
    histogram(X, Y)
    pie_chart(Y)
