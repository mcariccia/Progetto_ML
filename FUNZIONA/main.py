import tkinter as tk
import pandas as pd
import numpy as np
from tkinter import Toplevel, scrolledtext, ttk
from addestramenti import AddestramentoClassificatori
from dataset import Dataset
from dataAnalysis import data_analysis

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning App")
        self.root.geometry("600x600")
        self.root.resizable(False, False)

        # Terminale di output
        self.output_terminal = scrolledtext.ScrolledText(root, height=10, width=70, state='disabled')
        self.output_terminal.pack(pady=20)

        # Pulsante per il preprocessing
        self.preprocess_button = tk.Button(root, text="Preprocessing", command=self.open_preprocessing_window)
        self.preprocess_button.pack(pady=20)

        # Pulsante per i classificatori
        self.classifiers_button = tk.Button(root, text="Classificatori", command=self.open_classifiers_window)
        self.classifiers_button.pack(pady=20)

        # Pulsante per l'analisi dei dati
        self.data_analysis_button = tk.Button(root, text="Analisi dei Dati", command=self.open_data_analysis_window)
        self.data_analysis_button.pack(pady=20)

        self.df = pd.read_csv("world_population.csv")
        self.x = self.df[['2022 Population','2020 Population','2015 Population','2010 Population','2000 Population','1990 Population','1980 Population','1970 Population', 'Area (km²)', 'Density (per km²)', 'Growth Rate', 'World Population Percentage']]
        self.y = self.df['Continent']
        self.data = Dataset(self.x, self.y)

    def open_preprocessing_window(self):
        self.preprocess_window = Toplevel(self.root)
        self.preprocess_window.title("Preprocessing")
        self.preprocess_window.geometry("400x400")
        self.preprocess_window.resizable(False, False)

        varSelezioneFeatures = tk.StringVar()
        sceltaSelezioneFeatures = ["Nessuna Selezione", "Chi-Squared Selection", "Mutual Information", "F-Classif"]
        dropdownSelezioneFeatures = ttk.Combobox(self.preprocess_window, textvariable=varSelezioneFeatures, values=sceltaSelezioneFeatures, state="readonly")
        dropdownSelezioneFeatures.pack(pady=5)
        dropdownSelezioneFeatures.current(0)

        varBilanciamento = tk.StringVar()
        sceltaBilanciamento = ["Nessun Bilanciamento", "SMOTE", "Random Over Sampling", "Random Under Sampling"]
        dropdownBilanciamento = ttk.Combobox(self.preprocess_window, textvariable=varBilanciamento, values=sceltaBilanciamento, state="readonly")
        dropdownBilanciamento.pack(pady=5)
        dropdownBilanciamento.current(0)

        varStandardizzazione = tk.BooleanVar(value=True)
        iStandardizzazione = tk.Checkbutton(self.preprocess_window, text="Standardizzazione", variable=varStandardizzazione)
        iStandardizzazione.pack(pady=5)

        varNormalizzazione = tk.BooleanVar(value=True)
        iNormalizzazione = tk.Checkbutton(self.preprocess_window, text="Normalizzazione", variable=varNormalizzazione)
        iNormalizzazione.pack(pady=5)

        varRimozioneOutlier = tk.BooleanVar(value=True)
        iRimozioneOutlier = tk.Checkbutton(self.preprocess_window, text="Rimozione Outlier", variable=varRimozioneOutlier)
        iRimozioneOutlier.pack(pady=5)

        pulsanteApplica = tk.Button(self.preprocess_window, text="Applica", command=lambda: self.apply_preprocessing(varBilanciamento, varStandardizzazione, varNormalizzazione, varSelezioneFeatures, varRimozioneOutlier))
        pulsanteApplica.pack(pady=5)

        self.write_to_terminal("Preprocessing window opened")

    def apply_preprocessing(self, varBilanciamento, varStandardizzazione, varNormalizzazione, varSelezioneFeatures, varRimozioneOutlier):
        # Esegui il preprocessing utilizzando il metodo della classe Dataset
        self.data.preprocessing(
            varBilanciamento.get(),
            varStandardizzazione.get(),
            varNormalizzazione.get(),
            varSelezioneFeatures.get(),
            varRimozioneOutlier.get()
        )
        self.write_to_terminal("Preprocessing applicato.")
        self.write_data_info()
        self.preprocess_window.destroy()  # Chiudi la finestra di preprocessing
        self.root.focus()

    def open_classifiers_window(self):
        self.finestraClassificatore = Toplevel(self.root)
        self.finestraClassificatore.title("Addestra Classificatore")
        self.finestraClassificatore.geometry("400x400")

        tk.Label(self.finestraClassificatore, text="Scegli il Classificatore:").pack(pady=10)
        variabileClassificatore = tk.StringVar()
        scelteClassificatore = ["svm", "Naive Bayes", "kmeans", "ensemble", "knn Custom"]
        dropdownClassificatore = ttk.Combobox(self.finestraClassificatore, textvariable=variabileClassificatore, values=scelteClassificatore, state="readonly")
        dropdownClassificatore.pack(pady=5)
        dropdownClassificatore.current(0)

        varCross = tk.BooleanVar(value=True)
        iCross = tk.Checkbutton(self.finestraClassificatore, text="Cross Validation", variable=varCross)
        iCross.pack(pady=5)

        varTuning = tk.BooleanVar(value=True)
        iTuning = tk.Checkbutton(self.finestraClassificatore, text="Tuning", variable=varTuning)
        iTuning.pack(pady=5)

        tk.Label(self.finestraClassificatore, text="Calcolo della Distanza:").pack(pady=5)
        variabileMetrica = tk.StringVar()
        scelteMetrica = ["manhattan", "euclidean", "chebyshev"]
        dropdownMetrica = ttk.Combobox(self.finestraClassificatore, textvariable=variabileMetrica, values=scelteMetrica, state="readonly")
        dropdownMetrica.pack(pady=5)
        dropdownMetrica.current(0)

        tk.Label(self.finestraClassificatore, text="Valore di k:").pack(pady=5)
        variabileK = tk.StringVar(value="0")
        entryK = tk.Entry(self.finestraClassificatore, textvariable=variabileK)
        entryK.pack(pady=5)

        pulsanteAddestra = tk.Button(self.finestraClassificatore, text="Addestra",
            command=lambda: self.addestra(
                self.data.x_train,
                self.data.x_test,
                self.data.y_train,
                self.data.y_test,
                variabileClassificatore.get(),
                varCross.get(),
                varTuning.get(),
                variabileMetrica.get(),
                variabileK.get()
            ))
        pulsanteAddestra.pack(pady=20)

    def addestra(self, x_train, x_test, y_train, y_test, classificatore, cross_validation, tuning, metrica, k):
        # Addestra il classificatore utilizzando il metodo della classe AddestramentoClassificatori
        accuracy = AddestramentoClassificatori(
            x_train,
            x_test,
            y_train,
            y_test,
            classificatore,
            cross_validation,
            tuning,
            metrica,
            int(k)
        )
        self.write_to_terminal("Addestramento completato.")
        self.write_to_terminal(f"Accuratezza del modello {classificatore}: {accuracy:.2f}")
        self.finestraClassificatore.destroy()  # Chiudi la finestra di addestramento

    def open_data_analysis_window(self):
        data_analysis()

    def write_to_terminal(self, message):
        self.output_terminal.config(state='normal')
        self.output_terminal.insert(tk.END, message + '\n')
        self.output_terminal.config(state='disabled')

    def write_data_info(self):
        if self.data.x_train is not None:
            self.write_to_terminal("dai ti prego: " + str(self.data.x_train))
        else:
            self.write_to_terminal("dai ti prego: x_train è None")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
