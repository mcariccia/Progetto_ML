import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


df=pd.read_csv("world_population.csv",encoding='UTF-8') #importiamo dataset

#dividiamo il dataframe
df_y=df.loc[:,'Continent']
df_X=df.iloc[:, 5:]

#convertiamo in numpy
X=df_X.to_numpy()
y=df_y.to_numpy()


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



def train_test_knn(training_x, training_y, test_x, test_y): #funzione che crea, addestra e testa un KNN
    kNN_clf = KNeighborsClassifier(1)
    kNN_clf.fit(training_x, training_y)
    pred_y = kNN_clf.predict(test_x)
    return compute_accuracy(test_y, pred_y)

def compute_accuracy(pred_y, test_y):  #funzione per il calcolo dell'accuratezza del modello
    return (pred_y == test_y).sum() / len(pred_y)

training_x, test_x, training_y, test_y = train_test_split(X, y, random_state=0, test_size=0.25, stratify=y)  #dividiamo dataframe in training set e test set


#inizializziamo le liste che conterranno la performance di ogni metodo di riduzione e per ogni dimensione passata al metodo
knn_acc_srp = []  
knn_acc_grp = []
knn_acc_fa = []
knn_acc_pca = []
knn_acc_skb = [] 
knn_acc_skb2 = []
knn_acc_skb3 = [] 
knn_acc_sfs = [] 




#iteriamo per testare la performance del KNN dopo ogni metodo di riduzione e per ogni possibile numero di attributi rimanenti nel dataframe a seguito della riduzione
for dim in range(1,X.shape[1]):
    srp = SparseRandomProjection(dim, random_state=0)
    train_x_red = srp.fit_transform(training_x)   
    test_x_red = srp.transform(test_x)
    knn_acc_srp = knn_acc_srp + [train_test_knn(train_x_red, training_y, test_x_red, test_y)]
        

    grp = GaussianRandomProjection(dim, random_state=0)
    train_x_red = grp.fit_transform(training_x)
    test_x_red = grp.transform(test_x)
    knn_acc_grp = knn_acc_grp + [train_test_knn(train_x_red, training_y, test_x_red, test_y)]
        
        
    fa = FeatureAgglomeration(dim)
    train_x_red = fa.fit_transform(training_x)
    test_x_red = fa.transform(test_x)
    knn_acc_fa = knn_acc_fa + [train_test_knn(train_x_red, training_y, test_x_red, test_y)]
        
        
    pca = PCA(dim)
    train_x_red = pca.fit_transform(training_x)
    test_x_red = pca.transform(test_x)
    knn_acc_pca = knn_acc_pca + [train_test_knn(train_x_red, training_y, test_x_red, test_y)]        
          
    
    skb = SelectKBest(chi2, k=dim)
    train_x_red  = skb.fit_transform(training_x, training_y)
    test_x_red = skb.transform(test_x)
    knn_acc_skb = knn_acc_skb + [train_test_knn(train_x_red, training_y, test_x_red, test_y)]
        
      
    skb2 = SelectKBest(mutual_info_classif, k=dim)
    train_x_red  = skb2.fit_transform(training_x, training_y)
    test_x_red = skb2.transform(test_x)
    knn_acc_skb2 = knn_acc_skb2 + [train_test_knn(train_x_red, training_y, test_x_red, test_y)]
        
    
    skb3 = SelectKBestVar(dim) 
    train_x_red  = skb3.fit_transform(training_x)
    test_x_red = skb3.transform(test_x)  
    knn_acc_skb3 = knn_acc_skb3 + [train_test_knn(train_x_red, training_y, test_x_red, test_y)]
        


#inseriamo alla fine della lista l'accuratezza che si otterrebbe con tutti gli attributi
acc = train_test_knn(training_x, training_y, test_x, test_y)

knn_acc_srp = knn_acc_srp + [acc]
knn_acc_grp = knn_acc_grp + [acc]
knn_acc_fa = knn_acc_fa + [acc]
knn_acc_pca = knn_acc_pca + [acc]
knn_acc_skb = knn_acc_skb + [acc]
knn_acc_skb2 = knn_acc_skb2 + [acc]
knn_acc_skb3 = knn_acc_skb3 + [acc]

#calcoliamo la migliore dimensione per ogni metodo di riduzione sulla base del miglior risultato ottenuto
BestDim_srp=knn_acc_srp.index(max(knn_acc_srp))
BestDim_grp=knn_acc_grp.index(max(knn_acc_grp))
BestDim_fa=knn_acc_fa.index(max(knn_acc_fa))
BestDim_pca=knn_acc_pca.index(max(knn_acc_pca))
BestDim_skb=knn_acc_skb.index(max(knn_acc_skb))
BestDim_skb2=knn_acc_skb2.index(max(knn_acc_skb2))
BestDim_skb3=knn_acc_skb3.index(max(knn_acc_skb3))

#mostriamo i risultati
print(f'Migliori dimensioni per ogni funzione di riduzione dimensionalità:\nSparseRandomProjection=: {BestDim_srp}\nGaussianRandomProjection: {BestDim_grp}\nFeatureAgglomeration: {BestDim_fa}\nPrincipalComponentsAnalisys: {BestDim_pca}\nScoring_chi2: {BestDim_skb}\nScoring_mutual_information: {BestDim_skb2}\nVarianceThreshold: {BestDim_skb3}\n')
