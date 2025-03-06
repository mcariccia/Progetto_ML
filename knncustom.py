import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class KnnCustom:
    def __init__(self, x_train, y_train, param_grid, k_folds=5):
        # Conversione dei dati in array NumPy
        self.x_train = np.array(x_train, dtype=np.float64)
        self.y_train = np.array(y_train)
        self.param_grid = param_grid
        self.k_folds = k_folds

    # Metodo per calcolare la distanza tra i punti
    def calcolo_distanza(self, x_train, x_test, distanza):
        valdist = []
        x_train = np.array(x_train, dtype=np.float64)
        x_test = np.array(x_test, dtype=np.float64)
        for x_t in x_test:
            if distanza == 'euclidean':
                dist = np.sqrt(np.sum(np.square(x_train - x_t), axis=1))
            elif distanza == "manhattan":
                dist = np.sum(np.abs(x_train - x_t), axis=1)
            elif distanza == "chebyshev":
                dist = np.max(np.abs(x_train - x_t), axis=1)
            valdist.append(dist)
        return np.array(valdist)

    # Metodo per eseguire il KNN
    def knn(self, x_train, y_train, x_test, param):
        valdist = self.calcolo_distanza(x_train, x_test, param['distanza'])
        predictions = []
        y_train = np.array(y_train)  # Conversione di y_train in array NumPy
        for dist in valdist:
            nearest_indices = np.argsort(dist)[:param['k']]
            nearest_labels = y_train[nearest_indices]

            if param['peso'] == 'uniform':
                unique, counts = np.unique(nearest_labels, return_counts=True)
            elif param['peso'] == 'distance':
                weights = 1 / (dist[nearest_indices] + 1e-5)
                all_classes = np.unique(y_train)
                weighted_counts = {label: 0 for label in all_classes}
                for label, weight in zip(nearest_labels, weights):
                    weighted_counts[label] += weight
                unique = np.array(list(weighted_counts.keys()))
                counts = np.array(list(weighted_counts.values()))

            majority_vote = unique[np.argmax(counts)]
            predictions.append(majority_vote)
        return np.array(predictions)

    # Metodo per calcolare l'accuratezza
    def calcola_accuratezza(self, predictions, y_test):
        return np.sum(predictions == y_test) / len(y_test)

    # Metodo per eseguire il tuning degli iperparametri con cross-validation
    def tuning_con_cross_validation(self):
        fold_size = len(self.x_train) // self.k_folds
        fold_accuracies = []
        all_predictions = np.zeros(len(self.x_train), dtype=self.y_train.dtype)

        # Suddividi i dati in k fold
        for i in range(self.k_folds):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i != self.k_folds - 1 else len(self.x_train)
            X_val = self.x_train[val_start:val_end]
            y_val = self.y_train[val_start:val_end]
            X_train_fold = np.concatenate((self.x_train[:val_start], self.x_train[val_end:]), axis=0)
            y_train_fold = np.concatenate((self.y_train[:val_start], self.y_train[val_end:]), axis=0)

            best_accuracy = 0
            best_param = None

            # Testa ogni combinazione di iperparametri
            for k in self.param_grid['k']:
                for distanza in self.param_grid['distanza']:
                    for peso in self.param_grid['peso']:
                        param = {'k': k, 'distanza': distanza, 'peso': peso}
                        predictions = self.knn(X_train_fold, y_train_fold, X_val, param)
                        accuracy = self.calcola_accuratezza(predictions, y_val)

                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_param = param

            # Usa il miglior iperparametro per questo fold e calcola l'accuratezza su di esso
            predictions = self.knn(X_train_fold, y_train_fold, X_val, best_param)
            final_accuracy = self.calcola_accuratezza(predictions, y_val)
            fold_accuracies.append(final_accuracy)
            all_predictions[val_start:val_end] = predictions

        # Media delle accuratezze sui fold
        return np.mean(fold_accuracies)
