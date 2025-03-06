from sklearn import svm, metrics
from sklearn.model_selection import  GridSearchCV
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from ensembleClassifierGUI import ensemble_classifier
from knncustom import KnnCustom


def AddestramentoClassificatori(x_train, x_test, y_train, y_test, classificatore):
    if classificatore == "SVM":
        # 
        """param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
        }

        best_params = TuningIperparametri(svm.SVC(), param_grid, x_train, y_train)
        print(f"Migliori iperparametri: {best_params}")  """ 
        
        best_clf = svm.SVC(C=100, gamma=1, kernel='rbf')
        best_clf.fit(x_train, y_train)
        y_predv = best_clf.predict(x_test)
        test_accuracy = accuracy_score(y_test, y_predv)
        print(f"Accuratezza sul test set: {test_accuracy:.4f}")
        
        cv_scores = cross_val_score(best_clf, x_train, y_train, cv=5)
        print(f"Score della cross-validation: {cv_scores}")
        print(f"Accuratezza media CV: {cv_scores.mean():.4f}")
        accuracy = cv_scores.mean()

        print(f"Accuratezza del modello: {accuracy:.2f}")
        print("Modello addestrato")

    elif classificatore == "Naive Bayes":
        param_grid = {
            'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
            'norm': [True, False]
        }
        
        best_params = TuningIperparametri(ComplementNB(), param_grid, x_train, y_train)
        
        # Addestramento del modello con i migliori parametri
        best_clf = ComplementNB(**best_params)      # rimuovere l'argomento **best_params per usare i parametri di default
        best_clf.fit(x_train, y_train)
        y_predv = best_clf.predict(x_test)
        test_accuracy = accuracy_score(y_test, y_predv)
        print(f"Accuratezza sul test set: {test_accuracy:.4f}")
        
        cv_scores = cross_val_score(best_clf, x_train, y_train, cv=5)
        print(f"Score della cross-validation: {cv_scores}")
        print(f"Accuratezza media CV: {cv_scores.mean():.4f}")
        accuracy = cv_scores.mean()

        print(f"Accuratezza del modello: {accuracy:.2f}")
        print("Modello addestrato")


    elif classificatore == "Ensemble classifier":
        y_predv, accuracy = ensemble_classifier(x_train, x_test, y_train, y_test)
        
    elif classificatore == "Decision Tree":
        param_grid = {
            'max_depth': list(range(1, 25)),
            'max_leaf_nodes': list(range(2, 25)),
        }      
        best_params = TuningIperparametri(DecisionTreeClassifier(), param_grid, x_train, y_train)
        
        # Addestramento del modello con i migliori parametri
        best_clf = DecisionTreeClassifier(**best_params)    # rimuovere l'argomento **best_params per usare i parametri di default
        best_clf.fit(x_train, y_train)
        y_predv = best_clf.predict(x_test)
        test_accuracy = accuracy_score(y_test, y_predv)
        print(f"Accuratezza sul test set: {test_accuracy:.4f}")
        
        cv_scores = cross_val_score(best_clf, x_train, y_train, cv=5)
        print(f"Score della cross-validation: {cv_scores}")
        print(f"Accuratezza media CV: {cv_scores.mean():.4f}")
        accuracy = cv_scores.mean()
        
        print(f"Accuratezza del modello: {accuracy:.2f}")
        print("Modello addestrato")
        
    elif classificatore == "KNN Custom":
        param_grid = {
            'k': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'distanza': ['euclidean', 'manhattan', 'chebyshev'],
            'peso': ['uniform', 'distance']
        }

        # Inizializza il modello KNN
        knn_model = KnnCustom(x_train, y_train, param_grid, k_folds=5)

        # Esegui la cross-validation con tuning
        accuracy = knn_model.tuning_con_cross_validation()
        print(f"Accuratezza Media sui Fold con i Migliori Parametri: {accuracy}")
        
        # Genera le predizioni per l'intero set di test utilizzando i migliori parametri
        best_params = {
            'k': param_grid['k'][0],
            'distanza': param_grid['distanza'][0],
            'peso': param_grid['peso'][0]
        }
        
        y_predv = knn_model.knn(x_train, y_train, x_test, best_params)
            
    return y_predv, accuracy

def TuningIperparametri(classificatore, param_grid, x_train, y_train):
    grid_search = GridSearchCV(classificatore, param_grid, cv=5, refit=True, verbose=1, scoring="accuracy", n_jobs=-1)
    grid_search.fit(x_train, y_train)
    print(f"Migliori parametri trovati: {grid_search.best_params_}")
    print(f"Miglior score in CV: {grid_search.best_score_:.4f}")
    return grid_search.best_params_
