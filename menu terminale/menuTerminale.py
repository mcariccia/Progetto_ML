from dataAnalysis import data_analysis, load_data, feat_label
from ensembleClassifier import ensemble_classifier
from clusteringkmeans_mod import cluster
from Classificatore_KNN_custom import knn
from bayesian_classifier import raw_naiveBayes
from svm import svm_class
from data_standardization import standardization
from bilanciamento import apply_undersampling, apply_oversampling

# stampa le opzioni disponibili
def print_options():
    print("\nScegliere tra le seguenti funzioni:")
    print("1. Analisi")
    print("2. Classificazione")
    print("3. Pre-processing")
    print("4. Esci dal programma")
    
def clf(X, Y):
    print("\nClassificatori disponibili:")
    print("1. Kmeans")
    print("2. Classificatore multiplo custom")
    print("3. Classificatore KNN custom")
    print("4. Naive Bayes")
    print("5. SVM")
    
    clf_choice = int(input("\nInserire la scelta: "))
        
    if clf_choice == 1:
        cluster(X)
    elif clf_choice == 2:
        ensemble_classifier(X, Y)
    elif clf_choice == 3:
        knn(X, Y)
    elif clf_choice == 4:
        raw_naiveBayes(X, Y)
    elif clf_choice == 5:
        svm_class(X, Y)
    
def print_options_pre_proc():
    print("\nTipi di pre-processing disponibili:")
    print("1. Standardizzazione - Z-score")
    print("2. Standardizzazione - Minmax")
    print("3. Standardizzazione - Robust")
    print("4. Undersampling - Random")
    print("5. Undersampling - IHT")
    print("6. Oversampling - Random")
    print("7. Oversampling - SMOTE")
    
def menu(dataset):
    X, Y = feat_label(dataset)
    
    print_options()
    func_choice = int(input("\nInserire la scelta: "))
    
    if func_choice == 1:
        data_analysis()
        
    elif func_choice == 2:
        clf(X, Y)
        
            
    elif func_choice == 3:
        print_options_pre_proc()
        pre_choice = int(input("\nInserire la scelta: "))
        
        if pre_choice == 1:
            X_proc, Y_proc, processor = standardization(dataset, 'Continent', categorical_encoding='None', scaler_type='standard')
            X_proc.rename(columns = {list(X_proc)[-1]:'Capital', list(X_proc)[-2]:'Country/Territory', list(X_proc)[-3]:'CCA3'}, inplace=True)
        elif pre_choice == 2:
            X_proc, Y_proc, processor = standardization(dataset, 'Continent', categorical_encoding='None', scaler_type='minmax')
            X_proc.rename(columns = {list(X_proc)[-1]:'Capital', list(X_proc)[-2]:'Country/Territory', list(X_proc)[-3]:'CCA3'}, inplace=True)
        elif pre_choice == 3:
            X_proc, Y_proc, processor = standardization(dataset, 'Continent', categorical_encoding='None', scaler_type='robust')
            X_proc.rename(columns = {list(X_proc)[-1]:'Capital', list(X_proc)[-2]:'Country/Territory', list(X_proc)[-3]:'CCA3'}, inplace=True)
            
        elif pre_choice == 4:
            X_proc, Y_proc = apply_undersampling(X, Y, method="random")
        elif pre_choice == 5:
            X_proc, Y_proc = apply_undersampling(X, Y, method="iht")
        elif pre_choice == 6:
            X_proc, Y_proc = apply_oversampling(X, Y, method="random")
        elif pre_choice == 7:
            X_proc, Y_proc = apply_oversampling(X, Y, method="smote")
        
            
        
        clf(X_proc, Y_proc)
            
dataset = load_data()
menu(dataset)