import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder

#funzione per caricare i dati dal file
def load_data():
    #individua la posizione del file corrente - necessario per il percorso relativo
    path = Path(__file__).parent
    
    #crea un dataframe con il dataset
    dataset = pd.read_csv((path/"world_population.csv").resolve())
    #ridenominazione di attributi con nomi problematici
    dataset.rename(columns = {list(dataset)[-4]:'Area', list(dataset)[-3]:'Density'}, inplace=True)
    
    return dataset

#funzione per estrarre attributi e classe
def feat_label (dataset):
    
    #estrae gli attributi
    X_features = dataset.drop(labels=['Continent'], axis=1)
    #estrae la classe
    Y_label = dataset['Continent']
    
    #restituisce attributi e classe
    return X_features, Y_label

#funzione per l'analisi del dataset
def analysis (X):
    #estrazione dei nomi delle colonne
    feat_names = X.columns
    
    #identificazione e gestione delle colonne categoriche:
    #gli attributi categorici vengono convertiti in numerici
    X_numeric = X.copy() #creazione di una copia del dataframe per non perdere l'originale
    encoder = LabelEncoder() #creo l'encoder 
    for feat in feat_names: #ciclo per scorrere la lista delle features
        if type(X_numeric[feat][0]) == type("H"): #controllo se il primo elemento è di tipo stringa
            X_numeric[feat] = encoder.fit_transform(X_numeric[feat]) #conversione 
    
    #inizializzazione degli array
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
    
    #ciclo per scorrere gli attributi - calcolando i valori per ognuno
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
    
    
    #calcolo della correlazione
    correlation = X_numeric.corr()
    
    #restituzione delle liste dei valori ottenuti
    return means, modes, medians, std, var, range, tenth_perc, twentyfifth_perc, fiftieth_perc, seventyfifth_perc, ninetieth_perc, correlation

#funzione per la stampa dei risultati ottenuti dalla precedente
def print_analysis(info):
    #label per colonne e righe
    feat_names = ['Rank', 'CCA3', 'Country/Territory', 'Capital', '2022 Population', '2020 Population', '2015 Population', '2010 Population',
       '2000 Population', '1990 Population', '1980 Population', '1970 Population', 'Area', 'Density', 'Growth Rate',
       'World Population Percentage']
    
    operation = ['Mean', 'Mode', 'Median', 'Standard Deviation', 'Variance', 'Range', '10%', '25%', '50%', '75%', '90%']
    
    #creazione del nuovo dataframe
    results = pd.DataFrame(info[:-1], columns=feat_names, index=operation)
    
    #stampa dei risultati
    print("\nRisultati dell'analisi dei dati:")
    print(results)
    
    print("\nCorrelazione tra gli attributi:")
    print(info[-1])

#funzione per creare i boxplot degli attributi numerici
def boxplot (X):
    #ricerca degli attributi non numerici
    feat_names = X.columns
    
    cat = [] #inizializzo lista vuota
    for feat in feat_names: #ciclo per scorrere la lista delle features
        if type(X[feat][0]) == type("H"): #controllo se il primo elemento è di tipo stringa
            cat = cat + [feat] #se si, aggiunto alla lista dei categorici
    feat_names = [feat for feat in feat_names if feat not in cat] #creazione di lista di attributi numerici
    
    #creazione dei subplot
    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(30, 15))
    ax = ax.flatten() #trasforma la matrice 3x5 in una lista
    
    #titolo dei grafici
    fig.suptitle('Boxplot degli attributi - Senza outliers', fontsize=20)
    
    #ciclo per creare i grafici degli attributi - senza outliers
    for i, feat in enumerate(feat_names):
        #adatta gli assi ai dati
        plt.sca(ax[i])
        
        #crea il plot senza outliers
        X[feat].plot(kind='box', patch_artist = True, showfliers=False, grid= True)

        #individua i percentili 
        q25 = np.percentile(X[feat], 25)
        q50 = np.percentile(X[feat], 50)
        q75 = np.percentile(X[feat], 75)
        q90= np.percentile(X[feat], 90)
        perc_val = [q25, q50, q75, q90]

        #disegna i percentili
        plt.scatter([1, 1, 1, 1], perc_val, marker='o', color='red', label='Percentiles')

        #etichetta per i percentili
        perc_name = ["25%", "50%", "75%", "90%"]
        for i, val in enumerate(perc_val):
            plt.text(1.1, val, f"{perc_name[i]}: {val:.2f}")

        #crea legenda per il grafico
        plt.legend(loc='upper left')
    
    #cancella grafici vuoti
    fig.delaxes(ax[-2])
    fig.delaxes(ax[-1])
    
    #mostra i risultati
    plt.show()
    
    #creazione dei subplot
    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(30, 15))
    ax = ax.flatten() #trasforma la matrice 3x5 in una lista
    
    #titolo dei grafici
    fig.suptitle('Boxplot degli attributi - Con outliers', fontsize=20)
    
    #grafici degli attributi con outliers
    for i, feat in enumerate(feat_names):
        #adatta gli assi ai dati
        plt.sca(ax[i])
        
        #crea il plot senza outliers
        X[feat].plot(kind='box', patch_artist = True, grid=True)
        
        #individua i quantili calcolati nell'analisi
        q10 = np.percentile(X[feat], 10)
        q25 = np.percentile(X[feat], 25)
        q50 = np.percentile(X[feat], 50)
        q75 = np.percentile(X[feat], 75)
        q90= np.percentile(X[feat], 90)
        perc_val = [q10, q25, q50, q75, q90]

        #disegna i quantili
        plt.scatter([1, 1, 1, 1, 1], perc_val, marker='o', color='red', label='Percentiles')

        #etichetta per i quantili
        perc_name = ["10%", "25%", "50%", "75%", "90%"]
        for i, val in enumerate(perc_val):
            plt.text(1.1, val, f"{perc_name[i]}: {val:.2f}")

        #crea legenda per il grafico
        plt.legend(loc='upper left')
    
    #cancella grafici vuoti
    fig.delaxes(ax[-2])
    fig.delaxes(ax[-1])
    
    #mostra i risultati
    plt.show()
  
#funzione per creare istogrammi degli attributi numerici  
def histogram (X, Y):
    #ricerca degli attributi non numerici
    feat_names = X.columns
    
    cat = [] #inizializzo lista vuota
    for feat in feat_names: #ciclo per scorrere la lista delle features
        if type(X[feat][0]) == type("H"): #controllo se il primo elemento è di tipo stringa
            cat = cat + [feat] #se si, aggiunto alla lista dei categorici
    feat_names = [feat for feat in feat_names if feat not in cat] #creazione di lista di attributi numerici
    
    #creazione dei subplot
    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(30, 30))
    ax = ax.flatten() #trasforma la matrice 3x5 in una lista
    
    #titolo dei grafici
    fig.suptitle('Istogramma degli attributi', fontsize=20)
    
    #ciclo per creare i grafici degli attributi - senza outliers
    for i, feat in enumerate(feat_names):
        #adatta gli assi ai dati
        plt.sca(ax[i])
        
        #creazione del plot - suddiviso per continent
        X[feat][Y == 'Asia']         .plot(kind='hist', density= False, bins= 20, alpha= 0.5, label='Asia')
        X[feat][Y == 'Africa']       .plot(kind='hist', density= False, bins= 20, alpha= 0.5, label='Africa')
        X[feat][Y == 'Europe']       .plot(kind='hist', density= False, bins= 20, alpha= 0.5, label='Europe')
        X[feat][Y == 'Oceania']      .plot(kind='hist', density= False, bins= 20, alpha= 0.5, label='Oceania')
        X[feat][Y == 'North America'].plot(kind='hist', density= False, bins= 20, alpha= 0.5, label='North America')
        X[feat][Y == 'South America'].plot(kind='hist', density= False, bins= 20, alpha= 0.5, label='South America')

        #crea legenda per il grafico
        plt.legend(loc='upper right')
        
        #aggiunge il titolo all'istogramma
        plt.title(feat, fontsize= 10)
    
    #cancella grafici vuoti
    fig.delaxes(ax[-2])
    fig.delaxes(ax[-1])
    
    #mostra i risultati
    plt.show()

#definizione della funzione per una stampa migliore delle percentuali nel grafico a torta
def pct_chart(pct, values):
    #riceve la percentuale dalla funzione del grafico 
    #riceve i dati di input del grafico
    #usa entrambi per calcolare il numero di elementi per ogni "fetta"
    number = int(pct/100 * sum(values))
    #restituisce la stringa che viene stampata su ciascuna fetta
    return f"{pct:.3f}%\n({number:d})"

#funzione per un grafico a torta delle classi
def pie_chart (Y):
    #calcola le occorrenze delle classi
    asia          = (Y[Y=='Asia'].count()         )
    africa        = (Y[Y=='Africa'].count()       )
    europe        = (Y[Y=='Europe'].count()       )
    oceania       = (Y[Y=='Oceania'].count()      )
    north_america = (Y[Y=='North America'].count())
    south_america = (Y[Y=='South America'].count())
    
    #crea lista dei valori e dei label
    values = [asia, africa, europe, oceania, north_america, south_america]
    label = ['Asia', 'Africa', 'Europe', 'Oceania', 'North America', 'South America']
    
    #crea il grafico    
    plt.pie(values, labels=label, autopct=lambda pct: pct_chart(pct, values))
    
    #aggiunge il titolo
    plt.title("Proporzioni delle classi nel dataset", fontsize= 20)
    
    #stampa il grafico
    plt.show()
   
#funzione unica per eseguire tutte le analisi 
def data_analysis ():
    #carica il dataset 
    dataset = load_data()
    #divide il dataset in attributi e classi
    X, Y = feat_label(dataset)
    #calcolo e stampa dei risultati dell'analisi matematica
    print_analysis(analysis(X))
    #stampa dei grafici
    boxplot(X)
    histogram(X, Y)
    pie_chart(Y)
    