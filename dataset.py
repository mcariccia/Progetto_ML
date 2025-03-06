from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np

class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def set_data(self, x_train, x_test, y_train, y_test, x, y):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.x = x
        self.y = y

    def preprocessing(self, balancing, scaler, featuresselection, outliersremoval):
        if featuresselection == "Chi-Squared Selection":
            selector = SelectKBest(chi2, k=5)
            self.x = selector.fit_transform(self.x, self.y)
        elif featuresselection == "Mutual Information":
            selector = SelectKBest(mutual_info_classif, k=5)
            self.x = selector.fit_transform(self.x, self.y)
        elif featuresselection == "F-Classif":
            selector = SelectKBest(f_classif, k=5)
            self.x = selector.fit_transform(self.x, self.y) 

        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42, stratify= self.y)

        if outliersremoval:
            z = np.abs(stats.zscore(x_train))
            x_train = x_train[(z < 3).all(axis=1)]
            y_train = y_train[(z < 3).all(axis=1)]
            print("Outlier rimossi")

        if scaler == 'Standardizzazione':
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            print("Caratteristiche standardizzate")

        if scaler == 'Normalizzazione':
            scaler = MinMaxScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            print("Caratteristiche normalizzate")   

        if balancing == "SMOTE":
            smote = SMOTE(random_state=42)
            x_train, y_train = smote.fit_resample(x_train, y_train)
            print("Distribuzione delle classi nel set di addestramento bilanciato")
        elif balancing == "Random Over Sampling":          
            ros = RandomOverSampler(random_state=42)
            x_train, y_train = ros.fit_resample(x_train, y_train)
            print("Distribuzione delle classi nel set di addestramento bilanciato")
        elif balancing == "Random Under Sampling":
            rus = RandomUnderSampler(random_state=42)
            x_train, y_train = rus.fit_resample(x_train, y_train)
            print("Distribuzione delle classi nel set di addestramento bilanciato")

        self.set_data(x_train, x_test, y_train, y_test, self.x, self.y)

        

    


            
