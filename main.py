import tkinter as tk
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from tkinter import Toplevel, scrolledtext, ttk
from addestramenti import AddestramentoClassificatori
from dataset import Dataset
from dataAnalysis import data_analysis

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning App")
        self.root.geometry("1000x800")  # Finestra principale più grande
        self.root.resizable(False, False)

        # Terminale di output più grande con caratteri ingranditi
        self.output_terminal = scrolledtext.ScrolledText(root, height=15, width=90, state='disabled', font=("Arial", 12))
        self.output_terminal.pack(pady=30)

        # Pulsanti più grandi con font aumentato e più spazio tra loro
        self.preprocess_button = tk.Button(root, text="Preprocessing", font=("Arial", 14, "bold"), width=25, height=2, command=self.open_preprocessing_window)
        self.preprocess_button.pack(pady=20)

        self.classifiers_button = tk.Button(root, text="Classificatori", font=("Arial", 14, "bold"), width=25, height=2, command=self.open_classifiers_window)
        self.classifiers_button.pack(pady=20)

        self.data_analysis_button = tk.Button(root, text="Analisi dei Dati", font=("Arial", 14, "bold"), width=25, height=2, command=self.open_data_analysis_window)
        self.data_analysis_button.pack(pady=20)

        self.df = pd.read_csv("world_population.csv")
        self.x = self.df[['2022 Population','2020 Population','2015 Population','2010 Population','2000 Population','1990 Population','1980 Population','1970 Population', 'Area (km²)', 'Density (per km²)', 'Growth Rate', 'World Population Percentage']]
        self.y = self.df['Continent']
        self.data = Dataset(self.x, self.y)

    def open_preprocessing_window(self):
        self.preprocess_window = Toplevel(self.root)
        self.preprocess_window.title("Preprocessing")
        self.preprocess_window.geometry("500x500")  # Finestra più grande
        self.preprocess_window.resizable(False, False)

        labelFeatures = tk.Label(self.preprocess_window, text="Scegli il tipo di Selezione di Features", font=("Arial", 12, "bold"))
        labelFeatures.pack(pady=15)
        varSelezioneFeatures = tk.StringVar()
        sceltaSelezioneFeatures = ["Nessuna Selezione", "Chi-Squared Selection", "Mutual Information", "F-Classif"]
        dropdownSelezioneFeatures = ttk.Combobox(self.preprocess_window, textvariable=varSelezioneFeatures, values=sceltaSelezioneFeatures, state="readonly", font=("Arial", 12))
        dropdownSelezioneFeatures.pack(pady=5)
        dropdownSelezioneFeatures.current(0)

        labelBilanciamento = tk.Label(self.preprocess_window, text="Scegli il tipo di Bilanciamento", font=("Arial", 12, "bold"))
        labelBilanciamento.pack(pady=15)
        varBilanciamento = tk.StringVar()
        sceltaBilanciamento = ["Nessun Bilanciamento", "SMOTE", "Random Over Sampling", "Random Under Sampling"]
        dropdownBilanciamento = ttk.Combobox(self.preprocess_window, textvariable=varBilanciamento, values=sceltaBilanciamento, state="readonly", font=("Arial", 12))
        dropdownBilanciamento.pack(pady=5)
        dropdownBilanciamento.current(0)

        labelScaler = tk.Label(self.preprocess_window, text="Scegli lo Scaler in base al classificatore", font=("Arial", 12, "bold"))
        labelScaler.pack(pady=15)
        varScaler = tk.StringVar()
        sceltaScaler = ["Nessuno Scaler", "Standardizzazione", "Normalizzazione"]
        dropdownScaler = ttk.Combobox(self.preprocess_window, textvariable=varScaler, values=sceltaScaler, state="readonly", font=("Arial", 12))
        dropdownScaler.pack(pady=5)
        dropdownScaler.current(0)

        varRimozioneOutlier = tk.BooleanVar(value=False)
        iRimozioneOutlier = tk.Checkbutton(self.preprocess_window, text="Rimozione Outlier", font=("Arial", 12), variable=varRimozioneOutlier)
        iRimozioneOutlier.pack(pady=10)

        pulsanteApplica = tk.Button(self.preprocess_window, text="Applica", font=("Arial", 14, "bold"), width=15, height=1, command=lambda: self.apply_preprocessing(varBilanciamento, varScaler, varSelezioneFeatures, varRimozioneOutlier))
        pulsanteApplica.pack(pady=15)

    def open_classifiers_window(self):
        self.finestraClassificatore = Toplevel(self.root)
        self.finestraClassificatore.title("Addestra Classificatore")
        self.finestraClassificatore.geometry("500x500")  # Più grande

        tk.Label(self.finestraClassificatore, text="Scegli il Classificatore:", font=("Arial", 12, "bold")).pack(pady=15)
        variabileClassificatore = tk.StringVar()
        scelteClassificatore = ["SVM", "Naive Bayes", "Decision Tree", "Ensemble classifier", "KNN Custom"]
        dropdownClassificatore = ttk.Combobox(self.finestraClassificatore, textvariable=variabileClassificatore, values=scelteClassificatore, state="readonly", font=("Arial", 12))
        dropdownClassificatore.pack(pady=5)
        dropdownClassificatore.current(0)
        
        pulsanteAddestra = tk.Button(self.finestraClassificatore, text="Addestra", font=("Arial", 14, "bold"), width=15, height=1, command=lambda: self.addestra(
            self.data.x_train,
            self.data.x_test,
            self.data.y_train,
            self.data.y_test,
            variabileClassificatore.get(),
        ))
        pulsanteAddestra.pack(pady=20)

    def open_data_analysis_window(self):
        data_analysis()

    def apply_preprocessing(self, varBilanciamento, varScaler, varSelezioneFeatures, varRimozioneOutlier):
        # Esegui il preprocessing utilizzando il metodo della classe Dataset
        self.data.preprocessing(
            varBilanciamento.get(),
            varScaler.get(),
            varSelezioneFeatures.get(),
            varRimozioneOutlier.get()
        )
        self.write_to_terminal("Preprocessing applicato.")
        self.preprocess_window.destroy()  # Chiudi la finestra di preprocessing
        self.root.focus()
        
    def addestra(self, x_train, x_test, y_train, y_test, classificatore):
        # Addestra il classificatore utilizzando il metodo della classe AddestramentoClassificatori
        if self.data.x_train is not None:
            y_pred, accuracy = AddestramentoClassificatori(
                x_train,
                x_test,
                y_train,
                y_test,
                classificatore
            )
            self.write_to_terminal("Addestramento completato.")
            self.write_to_terminal(f"Accuratezza del modello {classificatore}: {accuracy:.2f}")
            #self.write_to_terminal(classification_report(y_test, y_pred, zero_division= 0))    De-commentare per vedere il classification report
        else:
            self.write_to_terminal("Fai il preprocessing per caricare il Dataset, le impostazioni di default tengono i dati grezzi del Dataset")


        # Chiudi la finestra di addestramento dopo aver mostrato l'accuratezza
        self.write_to_terminal("----------------------------------------------------------------------------------------------------------------------------------------------------")
        self.finestraClassificatore.after(500, self.finestraClassificatore.destroy)  # Chiude la finestra dopo 0,5 secondi


    def write_to_terminal(self, message):
        self.output_terminal.config(state='normal')
        self.output_terminal.insert(tk.END, message + '\n')
        self.output_terminal.config(state='disabled')
    

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
    
