import tkinter as tk
from tkinter import messagebox, ttk, scrolledtext
import pandas as pd
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from addestramenti import AddestramentoClassificatori

df = pd.read_csv("world_population.csv")

# Creazione della finestra principale
window = tk.Tk()
window.geometry("600x600")
window.title("Menu Principale Machine Learning")
window.resizable(False, False)
window.configure(background="white")

# Creazione del terminale di output
output = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=70, height=20)
output.pack(padx=10, pady=10)

# Funzione per aggiornare il terminale di output
def update_output(text):
    output.insert(tk.END, text + '\n')
    output.see(tk.END)

# Funzioni per le operazioni di machine learning
def Win_Preprocessing():
    finestraPreprocessing = tk.Toplevel(window)
    finestraPreprocessing.title("PreProcessing")
    finestraPreprocessing.geometry("400x400")

    # Creazione delle etichette e delle variabili per i controlli
    varBilanciamento = tk.StringVar()
    sceltaBilanciamento = ["Nessun Bilanciamento", "SMOTE", "Random Over Sampling", "Random Under Sampling"]
    dropdownBilanciamento = ttk.Combobox(finestraPreprocessing, textvariable=varBilanciamento, values=sceltaBilanciamento, state="readonly")
    dropdownBilanciamento.pack(pady=5)
    dropdownBilanciamento.current(0)

    varStandardizzazione = tk.BooleanVar(value=True)
    iStandardizzazione = tk.Checkbutton(finestraPreprocessing, text="Standardizzazione", variable=varStandardizzazione)
    iStandardizzazione.pack(pady=5)

    varNormalizzazione = tk.BooleanVar(value=True)
    iNormalizzazione = tk.Checkbutton(finestraPreprocessing, text="Normalizzazione", variable=varNormalizzazione)
    iNormalizzazione.pack(pady=5)

    varSelezioneFeatures = tk.BooleanVar(value=True)
    iSelezioneFeatures = tk.Checkbutton(finestraPreprocessing, text="Selezione Features", variable=varSelezioneFeatures)
    iSelezioneFeatures.pack(pady=5)


def Win_Classificatori():
    finestraClassificatore = tk.Toplevel(window)
    finestraClassificatore.title("Addestra Classificatore")
    finestraClassificatore.geometry("400x400")

    #
    tk.Label(finestraClassificatore, text="Scegli il Classificatore:").pack(pady=10)
    variabileClassificatore = tk.StringVar()
    scelteClassificatore = ["svm", "Naive Bayes", "ann", "knn", "voting"]
    dropdownClassificatore = ttk.Combobox(finestraClassificatore, textvariable=variabileClassificatore, values=scelteClassificatore, state="readonly")
    dropdownClassificatore.pack(pady=5)
    dropdownClassificatore.current(0)

    varCross = tk.BooleanVar(value=True)
    iCross = tk.Checkbutton(finestraClassificatore, text="Cross Validation", variable=varCross)
    iCross.pack(pady=5)

    varTuning = tk.BooleanVar(value=True)
    iTuning = tk.Checkbutton(finestraClassificatore, text="Tuning", variable=varTuning)
    iTuning.pack(pady=5)

    #
    tk.Label(finestraClassificatore, text="Calcolo della Distanza:").pack(pady=5)
    variabileMetrica = tk.StringVar()
    scelteMetrica = ["manhattan", "euclidean", "chebyshev"]
    dropdownMetrica = ttk.Combobox(finestraClassificatore, textvariable=variabileMetrica, values=scelteMetrica, state="readonly")
    dropdownMetrica.pack(pady=5)
    dropdownMetrica.current(0)

    #
    tk.Label(finestraClassificatore, text="Valore di k:").pack(pady=5)
    variabileK = tk.StringVar(value="0")
    entryK = tk.Entry(finestraClassificatore, textvariable=variabileK)
    entryK.pack(pady=5)
    pulsanteAddestra = tk.Button(finestraClassificatore, text="Addestra",
        command=lambda: addestramento(
            variabileClassificatore.get(),
            varCross.get(),
            varTuning.get(),
            variabileMetrica.get(),
            variabileK.get(),
            finestraClassificatore
        ))
    pulsanteAddestra.pack(pady=20)

    def addestramento(classificatore, cross_validation, tuning, distanza, k, finestra):
        # Converte k in intero
        valk = int(k)
        # Chiama la funzione classifiers passando anche i parametri 'distanza' e 'kValue'
        y_pred, accuracy = AddestramentoClassificatori(df, classificatore, cross_validation, tuning, distanza, valk)

        update_output(f"L'accuratezza per l'algoritmo {classificatore} con le impostazioni inserite per l'addestramento è {accuracy}")
    
        
        finestra.destroy()

def Win_AnalisiDati():
    update_output("Esecuzione dell'analisi dei dati...")

# Creazione dei pulsanti
ButtonPreProcessing = tk.Button(window, text="PreProcessing", command=Win_Preprocessing)
ButtonPreProcessing.pack(padx=20, pady=20)

ButtonClassificatori = tk.Button(window, text="Classificatori", command=Win_Classificatori)
ButtonClassificatori.pack(padx=20, pady=20)

ButtonAnalisi = tk.Button(window, text="Analisi dei Dati", command=Win_AnalisiDati)
ButtonAnalisi.pack(padx=20, pady=20)

# Avvio del loop principale di Tkinter
if __name__ == "__main__":
    window.mainloop()
