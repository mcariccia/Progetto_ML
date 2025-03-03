import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold


class SelectKBestVar:
    def __init__(self, k):
        self.k = k
        self.selector= None
        
    def fit_transform(self, X):
        self.var = np.sort(np.var(X, axis=0))[::-1]  #ordiniamo la varianza di ogni attributo in senso decrescente
        self.selector = VarianceThreshold(threshold=self.var[self.k])  #selezioniamo come varianza quella del k-esimo elemento
        X_reduced = self.selector.fit_transform(X)  #eliminiamo gli attributi con varianza minore di quella del k-esimo elemento
        return X_reduced
    
    def transform(self, X):
        X_reduced = self.selector.transform(X) #operazione effettuata sul test set
        return X_reduced


def compute_accuracy(pred_y, test_y):  #funzione per il calcolo dell'accuratezza del modello
    return (pred_y == test_y).sum() / len(pred_y)


df=pd.read_csv("world_population.csv",encoding='UTF-8') #importiamo dataset

#dividiamo il dataframe
df_y=df.loc[:,'Continent']
df_X=df.iloc[:, 5:]

#convertiamo in numpy
X=df_X.to_numpy()
y=df_y.to_numpy()

def riduzione_dimensionalità(X,y,df,tipo):

    if tipo==0:  #Principal Component Analysis
        pca = PCA(2)
        X_reduced = pca.fit_transform(X)

    
    #metodi di selezione delle features
    elif tipo==1:
        
        corr_df=df.corr()
        correlazione=np.where(corr_df.to_numpy()>0.93)   #individuiamo tutte le coppie di attributi che hanno un valore di correlazione >0.93

        lista_correlati=[]  #lista che contiene gli attuali indici degli attributi correlati all'attributo il cui indice è attaulmente in posizione [0][i]
        lista_scartati=[]  #lista che contiene l'indice di tutti quegli attributi correlati scartati al fine di sceglierne solo uno
        indici_features=[] #contiene gli effettivi indici degli attributi correlati che abbiamo scelto di mantenere

        #algoritmo per l'individuazione degli attributi correlati da mantenere
        for i in range(len(correlazione[0])): #iteriamo per ogni indice nel primo array di 'correlazione'
    
            if i!=0 and correlazione[0][i]!=correlazione[0][i-1] and correlazione[0][i-1] not in lista_scartati:    #condizione che si attiva se l attributo considerato in [0][i] cambia e se esso non appartiene alla lista degli scartati
                    indici_features.append(correlazione[0][i-1])      #inseriamo l'indice dell' attributo tra quelli da mantenere
                    lista_scartati=lista_scartati+lista_correlati     #gli altri attributi correlati ad esso vengono scartati
                    lista_correlati=[]                                #riazzeriamo lista_correlati
                    
            if correlazione[0][i] not in lista_scartati and correlazione[0][i]!=correlazione[1][i]: # 
                lista_correlati.append(correlazione[1][i]) #inseriamo l'indice dell'attributo correlato
                
            if i==len(correlazione[0])-1 and correlazione[0][i-1] not in lista_scartati:   #questo si attiva all'ultima ripetizione per aggiungere l'ultimo indice se necessario
                indici_features.append(correlazione[0][i])
            

        X_reduced=X[:, indici_features] #selezioniamo da X tutte le righe e solo le colonne che abbiamo deciso di mantenere   
            
    
    elif tipo==2:  #Scoring_MutualInfo
        X_reduced  = SelectKBest(mutual_info_classif, k=6).fit_transform(X, y)  #seleziona le prime k migliori caratteristiche sulla base dell'informazione mutua
        
    
    return X_reduced



X_reduced=riduzione_dimensionalità(X,y,df_X,i)