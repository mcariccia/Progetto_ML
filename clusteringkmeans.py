
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid

# Caricamento dataset
file_path = "world_population.csv"  # Modifica con il percorso corretto
df = pd.read_csv(file_path)

# Selezione delle colonne rilevanti per il clustering
pop_columns = ["1970 Population", "1980 Population", "1990 Population",
               "2000 Population", "2010 Population", "2015 Population",
               "2020 Population", "2022 Population"]

features = pop_columns + ["Area (km²)", "Density (per km²)", "Growth Rate"]
X = df[features]

# Normalizzazione dei dati per evitare distorsioni nei calcoli
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Funzione per trovare il miglior valore di K con SSE e Silhouette Score
def analyse_cluster(X, n_clusters_array):
    SSE = []
    Sil_score = []

    for k in n_clusters_array:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        SSE.append(kmeans.inertia_)
        labels = kmeans.predict(X)
        Sil_score.append(silhouette_score(X, labels))

    plt.figure(figsize=(12, 5))

    # Metodo del Gomito
    plt.subplot(1, 2, 1)
    plt.plot(n_clusters_array, SSE, marker='o', linestyle='--', color="darkorange")
    plt.xlabel("Numero cluster (K)")
    plt.ylabel("SSE (Inertia)")
    plt.title("Metodo del Gomito")

    # Coefficiente di Silhouette
    plt.subplot(1, 2, 2)
    plt.plot(n_clusters_array, Sil_score, marker='o', linestyle='--', color="darkorange")
    plt.xlabel("Numero cluster (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Analisi del Coefficiente di Silhouette")

    plt.show()

    return SSE, Sil_score

# Determino il miglior numero di cluster (K)
n_clusters_array = [2, 3, 4, 5, 8, 10, 12, 15, 20, 25]
SSE, Sil_score = analyse_cluster(X_scaled, n_clusters_array)


best_K = 6 

# Funzione per il tuning degli iperparametri 
def optimize_kmeans(X, best_K):
    param_grid = {
        "n_clusters": [best_K],
        "init": ["k-means++", "random"],
        "n_init": [10, 20, 50],
        "max_iter": [300, 500]
    }

    best_score = -1
    best_params = {}

    for params in ParameterGrid(param_grid):
        kmeans = KMeans(**params, random_state=42)
        labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, labels)

        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_params = params

    print("\n🔹 Migliori iperparametri trovati:", best_params)
    print("🔹 Miglior silhouette score:", best_score)

# Ottimizzo gli iperparametri per il valore ottimale di K
optimize_kmeans(X_scaled, best_K)

# Creo il modello finale con i parametri ottimizzati
final_kmeans = KMeans(n_clusters=best_K, init="k-means++", n_init=20, max_iter=300, random_state=42)
df["Cluster"] = final_kmeans.fit_predict(X_scaled)

# Calcolo il Silhouette Score per valutare il clustering
sil_score = silhouette_score(X_scaled, df["Cluster"])
print(f"\n🔹 Silhouette Score: {sil_score:.4f}")

# Visualizzo tutti i grafici in un'unica figura
plt.figure(figsize=(18, 6))

# Grafico 1: Metodo del Gomito
plt.subplot(1, 3, 1)
plt.plot(n_clusters_array, SSE, marker='o', linestyle='--', color="darkorange")
plt.axvline(x=best_K, linestyle="--", color="red", label=f"Best K: {best_K}")
plt.xlabel("Numero cluster (K)")
plt.ylabel("SSE (Inertia)")
plt.title("Metodo del Gomito")
plt.legend()

# Grafico 2: Silhouette Score
plt.subplot(1, 3, 2)
plt.plot(n_clusters_array, Sil_score, marker='o', linestyle='--', color="blue")
plt.axvline(x=best_K, linestyle="--", color="red", label=f"Best K: {best_K}")
plt.xlabel("Numero cluster (K)")
plt.ylabel("Silhouette Score")
plt.title("Analisi del Silhouette Score")
plt.legend()

# Grafico 3: Risultati del Clustering
plt.subplot(1, 3, 3)
sns.scatterplot(x=df["2022 Population"], y=df["Growth Rate"], hue=df["Cluster"], palette="viridis", alpha=0.7)
plt.xlabel("Popolazione 2022")
plt.ylabel("Tasso di Crescita (%)")
plt.title("Distribuzione dei Cluster: Popolazione vs Crescita")
plt.xscale("log")  # Scala logaritmica per migliorare la leggibilità
plt.legend(title="Cluster")

# Mostro tutti i grafici
plt.tight_layout()
plt.show()

 
