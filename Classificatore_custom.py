import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix


#funzione per il calcolo delle prestazioni in un problema multiclasse
def compute_performances_multiclass(cms):
    dim = len(cms)
    TP = np.empty(dim)
    TN = np.empty(dim)
    FP = np.empty(dim)
    FN = np.empty(dim)
    TPR = np.empty(dim)
    TNR = np.empty(dim)
    FPR = np.empty(dim)
    FNR = np.empty(dim)
    p = np.empty(dim)
    r = np.empty(dim)
    F1 = np.empty(dim)
    for i, cm in enumerate(cms):
        TP[i] = cm[1,1]
        TN[i] = cm[0,0]
        FP[i] = cm[0,1]
        FN[i] = cm[1,0]
        TPR[i] = TP[i] / (TP[i] + FN[i] + eps)
        TNR[i] = TN[i] / (TN[i] + FP[i] + eps)
        FPR[i] = FP[i] / (TN[i] + FP[i] + eps)
        FNR[i] = FN[i] / (TP[i] + FN[i] + eps)
        p[i] = TP[i] / (TP[i] + FP[i] + eps)
        r[i] = TPR[i]
        F1[i] = 2*r[i]*p[i] / (r[i] +p[i] + eps)
    
    return TPR.mean(), TNR.mean(), FPR.mean(), FNR.mean(), p.mean(), r.mean(), F1.mean()





class KNNCustom():

    def __init__(self,k ,dist): #con k = numero di elementi vicini da valutare per la predizione e dist = tipo di distanza che il classificatore utilizza per la predizione
        self.X = []
        self.y = []
        self.k=k
        self.dist=dist
        

    def fit(self, train_x, train_y):
        self.X = train_x
        self.y = train_y
        return

    
    def predict(self,test_x):  
        label_predette=[]     #lista che conterrà le label predette
        
        for test_record  in test_x:  #iteriamo su ogni record del test set
            distanze=[]   #lista che conterrà la distanza per ogni record del training set
            for training_record in self.X:  #iteriamo su ogni record del training set
                
                distanza= self.calcolo_distanza(training_record, test_record)    #calcoliamo distanza tra i due record
                
                distanze.append(distanza)     
        
            
           
            indici_elementi = np.argsort(distanze)[:self.k]  #qui con argosort andiamo a creare una lista di indici ordinata in base alla distanza e di lunghezza = k
            
            

            classi= self.y[indici_elementi]   #lista che contiene le effettive classi dei k elementi più vicini
            
            conteggio_classi=Counter(classi)  #contiamo quante volte si presenta ogni classe 
           
            classe_predetta = conteggio_classi.most_common(1)[0][0]  #con most_common andiamo a identificare la classe più comune, se due classi hanno la stessa frequenza sceglie quella che si presenta prima nella lista
           
            label_predette.append(classe_predetta)  #inseriamo la classe predetta nella lista
            
        return label_predette   

    
    def calcolo_distanza(self,r1,r2):
        
        if self.dist=='distanza_euclidea':
            
            return np.linalg.norm(r1-r2)  #funzione ottimizzata di numpy che ci permette di calcolare in modo efficiente la distanza euclidea tra i due vettori
            
        else:
            return np.sum(np.abs(r1 - r2)) #distanza di manhattan
    
            


    
    
def compute_accuracy(pred_y, test_y):     #funzione per il calcolo dell'accuratezza del classificatore
    return (pred_y == test_y).sum() / len(pred_y)



#------CARICAMENTO E SPLIT DEL DATASET-------#      

df=pd.read_csv("world_population.csv",encoding='UTF-8') #

y=df.loc[:,'Continent']
X=df.iloc[:, 5:]

y=y.to_numpy()
X=X.to_numpy()


training_x, test_x, training_y, test_y = train_test_split(X, y, random_state=0,test_size=0.25,stratify=y)  #dividiamo il dataset in training set e test set




#-----TUNING DEGLI IPERPARAMETRI--------#
#variabili per il tuning degli iperparametri
best_accuracy=-1      #accuracy massima ottenuta dai classificatori
best_k=-1             #numero di elementi vicini considerati nel classificatore con l'accuracy migliore
best_dist=-1          #tipo di distanza utilizzata dal classificatore con l'accuracy migliore
distanze=['distanza_euclidea','distanza_manhattan'] #lista delle distanze utilizzabili dal classificatore

for d in distanze:  #iteriamo sul tipo di distanza utilizzata
    for k in range(1,20):  #iteriamo sui possibili valori di k
        
        train_x, validation_x, train_y,  validation_y= train_test_split(training_x, training_y,random_state=0, test_size=0.25,stratify=training_y) #splittiamo il training set per ottenere un validation set

        #creiamo e addestriamo il classificatore
        clf=KNNCustom(k,d)  
        clf.fit(train_x,train_y)
        pred_y=clf.predict(validation_x)
        
        if compute_accuracy(pred_y,validation_y)>best_accuracy:   #verifichiamo che l'accuracy del classificatore attuale sia maggiore dell'accuracy di qualsiasi altro classificatore creato in precedenza
            best_accuracy=compute_accuracy(pred_y,validation_y)   #aggiorniamo best_accuracy e salviamo gli iperparametri con il quale abbiamo ottenuto questa performance
            best_k=k
            best_dist=d


#----VERIFICA DEI RISULTATI CON IPERPARAMETRI OTTIMIZZATI-------#

#una volta individuati i migliori iperparametri creiamo il classificatore ottimale 
best_clf=KNNCustom(best_k,best_dist)
best_clf.fit(training_x,training_y)
pred_y=best_clf.predict(test_x)


#----RISULTATI----#
#definiamo un valore di eps per evitare divisione per 0
eps = np.finfo(float).eps
cms = multilabel_confusion_matrix(test_y, pred_y, labels=np.unique(test_y)) #calcoliamo matrici di confusione per ogni classe
TPR,TNR,FPR,FNR,p,r,f1=compute_performances_multiclass(cms)  #calcoliamo le medie delle statistiche principali di performance di ogni classe

print(f'Migliori Iperparametri individuati: k={best_k} elementi vicini cosiderati e {best_dist} come distanza utilizzata')
print(f'L\'accuratezza è del {round(compute_accuracy(pred_y, test_y),2)}')
print()

#stampiamo le prestazioni medie in ordine TPR, TNR, FPR, FNR, p, r, F1
print('Più nello specifico le classi hanno ottenuto queste performance medie:')
print(f'TPR medio:{round(TPR,2)}\nTNR medio:{round(TNR,2)}\nFPR medio:{round(FPR,2)}\nFNR medio:{round(FNR,2)}\nprecision medio:{round(p,2)}\nrecall medio:{round(r,2)}\nf1 score medio:{round(f1,2)}\n')
