import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

features, true_labels = make_blobs(
    n_samples=2000,
    centers=3,
    cluster_std=2.75,
    random_state=42
)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=None
)

kmeans.fit(scaled_features)

temp_df = pd.DataFrame(scaled_features)

import seaborn as sns
sns.set_theme(style="white")

# Plot miles per gallon against horsepower with other semantics
sns.relplot(x=temp_df.iloc[:,0], y=temp_df.iloc[:,1], hue=kmeans.labels_,
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6)