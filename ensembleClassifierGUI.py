import pandas as pd
import numpy as np

from pathlib import Path

from statistics import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

path = Path(__file__).parent

dataset = pd.read_csv((path/"world_population.csv").resolve())

X_features = dataset.drop(labels=['CCA3', 'Capital', 'Continent', 'Country/Territory'], axis=1)
Y_label = dataset['Continent']

#definizione della classe per il classificatore multiplo
class Ensemble_classifier:
    #inizializzazione dei classificatori, il codice per il tuning degli iperparametri è in fondo al file
    def __init__(self):
        self.DTree = DecisionTreeClassifier(max_depth=7, max_leaf_nodes=10, min_impurity_decrease=0.0125)
        self.SVC = SVC(C=0.1, gamma=1)
        self.NBayes = GaussianNB()
     
    #fit dei classificatori   
    def fit(self, X_train, Y_train):
        self.DTree.fit(X_train, Y_train)
        self.SVC.fit(X_train, Y_train)
        self.NBayes.fit(X_train, Y_train)
    
    #predizione dei classificatori base   
    def base_pred(self, X_test):
        self.DTree_pred = self.DTree.predict(X_test)
        self.SVC_pred = self.SVC.predict(X_test)
        self.NBayes_pred = self.NBayes.predict(X_test)
    
    #hard voting per ottenere la predizione finale    
    def hard_voting(self):
        self.hard_pred = [mode([self.DTree_pred[i], self.SVC_pred[i], self.NBayes_pred[i]]) for i in range(len(self.DTree_pred))]
      
#funzione per stampare accuratezza, precisione, recall ed f1 di ogni classificatore nell'ensemble
#considerando il fatto che il problema è multiclasse, usa una media 'macro' per assegnare lo stesso peso a tutte le classi  
def evaluate_ensemble (ensemble, Y_test):
    print("Decision Tree Accuracy: ", accuracy_score(Y_test, ensemble.DTree_pred))
    print("Decision Tree Precision: ", precision_score(Y_test, ensemble.DTree_pred, average='macro', zero_division= 0))
    print("Decision Tree Recall: ", recall_score(Y_test, ensemble.DTree_pred, average='macro'))
    print("Decision Tree F1: ", f1_score(Y_test, ensemble.DTree_pred, average='macro'), "\n")
    
    print("Support Vector Accuracy: ", accuracy_score(Y_test, ensemble.SVC_pred))
    print("Support Vector Precision: ", precision_score(Y_test, ensemble.SVC_pred, average='macro', zero_division= 0))
    print("Support Vector Recall: ", recall_score(Y_test, ensemble.SVC_pred, average='macro'))
    print("Support Vector F1: ", f1_score(Y_test, ensemble.SVC_pred, average='macro'), "\n")
    
    print("Naive Bayes Accuracy: ", accuracy_score(Y_test, ensemble.NBayes_pred))
    print("Naive Bayes Precision: ", precision_score(Y_test, ensemble.NBayes_pred, average='macro', zero_division= 0))
    print("Naive Bayes Recall: ", recall_score(Y_test, ensemble.NBayes_pred, average='macro'))
    print("Naive Bayes F1: ", f1_score(Y_test, ensemble.NBayes_pred, average='macro'), "\n")
    
    print("Hard Voting Accuracy: ", accuracy_score(Y_test, ensemble.hard_pred))
    print("Hard Voting Precision: ", precision_score(Y_test, ensemble.hard_pred, average='macro', zero_division= 0))
    print("Hard Voting Recall: ", recall_score(Y_test, ensemble.hard_pred, average='macro'))
    print("Hard Voting F1: ", f1_score(Y_test, ensemble.hard_pred, average='macro'), "\n")
        
def ensemble_classifier (X_train, X_test, Y_train, Y_test):    
    #crea il classificatore
    ensemble_class = Ensemble_classifier()
    
    #addestra i classificatori base
    ensemble_class.fit(X_train, Y_train)
    
    #esegue le predizioni dei classificatori base
    ensemble_class.base_pred(X_test)
    
    #esegue il voto di maggioranza
    ensemble_class.hard_voting()
    
    #valuta i risultati ottenuti
    evaluate_ensemble(ensemble_class, Y_test)


#Tuning del decision tree:
#Prima eseguo un tuning individuale per 3 iperparametri: max_depth, max_leaf_nodes, min_impurity_decrease

def DTree_max_depth (X, Y):
    #divisione del dataset in training e test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=30, stratify=None)
    
    #creazione di una lista con tutti i valori considerati dal tuning
    max_depth_range = list(range(1, 25))
    acc = []
    
    #ciclo per testare i vari classificatori
    for depth in max_depth_range:
        clf = DecisionTreeClassifier(max_depth = depth, random_state = 0)
        clf.fit(X_train, Y_train)      
        #valutazione dell'accuracy score di orgni classificatore
        score = clf.score(X_test, Y_test)
        acc = acc + [score]
        #stampa dei risultati per il confronto
        print("Il valore di accuratezza per un albero di profondità ", depth, " è pari a : ", score)
#La max_depth migliore è 7

def DTree_max_leaf (X, Y):
    #divisione del dataset in training e test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=30, stratify=None)
    
    #creazione di una lista con tutti i valori considerati dal tuning
    max_leaf_nodes = list(range(2, 25))
    acc = []
    
    #ciclo per testare i vari classificatori
    for nodes in max_leaf_nodes:
        clf = DecisionTreeClassifier(max_leaf_nodes= nodes, random_state = 0)
        clf.fit(X_train, Y_train)    
        #valutazione dell'accuracy score di orgni classificatore  
        score = clf.score(X_test, Y_test)
        acc = acc + [score]
        #stampa dei risultati per il confronto
        print("Il valore di accuratezza per un albero con ", nodes, " foglie è pari a : ", score)
#La max_leaf_nodes migliore è 10

def DTree_min_gain (X, Y):
    #divisione del dataset in training e test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=30, stratify=None)
    
    #creazione di una lista con tutti i valori considerati dal tuning
    min_gain_val = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
    acc = []
    
    #ciclo per testare i vari classificatori
    for min_gain in min_gain_val:
        clf = DecisionTreeClassifier(min_impurity_decrease= min_gain, random_state = 0)
        clf.fit(X_train, Y_train)      
        #valutazione dell'accuracy score di orgni classificatore  
        score = clf.score(X_test, Y_test)
        acc = acc + [score]
        #stampa dei risultati per il confronto
        print("Il valore di accuratezza per un albero con guadagno minimo ", min_gain, " è pari a : ", score)
#La min_gain migliore è 0.025

#tuning del decision tree usando il gridsearch per valutare la combinazione migliore dei 3 iperparametri e del criterio di split
def DTree_gridsearch (X, Y):
    #divisione del dataset in training e test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=30, stratify=None)
    
    #definzione del dizionario con i parametri da valutare:
    param_dtree_tune = {'criterion'             : ['gini', 'entropy'], #criterio di split
                        'max_depth'             : list(range(1, 25)), #profondità massima
                        'max_leaf_nodes'        : list(range(2, 25)), #numero di foglie
                        'min_impurity_decrease' : [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]} #guadagno minimo
    
    #creazione del classificatore di test
    DTree_tune = DecisionTreeClassifier()
    
    #creazione dell'oggetto per la ricerca
    grid = GridSearchCV(DTree_tune, param_dtree_tune, scoring='accuracy')
    #valutazione delle possibili combinazioni
    grid.fit(X_train, Y_train)
    #ricerca della combinazione migliore
    print(grid.best_params_)
#I risultati migliori si hanno con: gini, depth 7 , leaf 10, min gain 0.0125

#tuning del support vector classifier con il gridsearch
def SVC_gridsearch (X, Y):
    #divisione del dataset in training e test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=30, stratify=None)
    
    #definzione del dizionario con i parametri da valutare:
    param_svc_tune = {'C'    : [0.1, 1, 10, 100, 1000], #peso dato alla variabile di slack nell'algoritmo
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001]} #variabile di slack
    
    #creazione del classificatore di test
    SVC_tune = SVC()
    
    #creazione dell'oggetto per la ricerca
    grid = GridSearchCV(SVC_tune, param_svc_tune, scoring='accuracy')
    #valutazione delle possibili combinazioni
    grid.fit(X_train, Y_train)
    #ricerca della combinazione migliore
    print(grid.best_params_)
#I risultati migliori si hanno con: C = 0.1 e gamma = 1
