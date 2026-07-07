## 🎵 Amazon-Music-Clustering
An unsupervised machine learning project that automatically categorizes music tracks into meaningful clusters based on their audio characteristics using K-Means and DBSCAN clustering algorithms.

### Project Overview

Amazon Music Clustering is a Machine Learning project that groups songs into meaningful clusters based on their audio characteristics. By applying unsupervised learning techniques, the project identifies patterns among songs and categorizes them into similar musical groups without predefined labels.

The primary objective is to analyze song features such as danceability, energy, tempo, acousticness, loudness, and valence to discover hidden structures within the music dataset.

---

### Problem Statement

Music streaming platforms contain thousands of songs with diverse characteristics. Manually categorizing songs is time-consuming and subjective. This project leverages clustering algorithms to automatically group songs with similar audio properties, helping improve:

* Music recommendation systems
* Playlist generation
* Genre discovery
* User listening experience
* Music analytics
  
---
  
### Dataset

The dataset contains information about songs and their audio features, such as:

* Danceability
* Energy
* Loudness
* Speechiness
* Acousticness
* Instrumentalness
* Liveness
* Valence
* Tempo
* Duration

Before clustering, song names, artist names, and IDs were removed because they are not useful for grouping songs.

---

### Features
* Data Exploration & Preprocessing: Comprehensive EDA, handling missing values, and feature scaling
* Multiple Clustering Algorithms: K-Means and DBSCAN implementation
* Optimal Cluster Selection: Elbow method and Silhouette score analysis
* Dimensionality Reduction: PCA visualization for 2D cluster representation
* Model Persistence: Trained models saved for production use
* Interactive Web App: Streamlit-based predictor for real-time cluster prediction
* Comprehensive Analysis: Cluster profiling and genre inference
---

### Tech Stack
**Languages & Libraries:**

* Python, Pandas, NumPy - Data analysis & manipulation
* Scikit-learn - Machine learning algorithms
* Matplotlib - Data visualization
* Joblib - Model serialization
* Streamlit - Web application framework

**Algorithms:**

* K-Means Clustering
* DBSCAN (Density-Based Spatial Clustering) 
* Principal Component Analysis (PCA)
* StandardScaler for feature normalization

---

### Project Workflow
**1. Data Loading**

The dataset is loaded into a Pandas DataFrame for analysis and preprocessing.

---
**2. Exploratory Data Analysis**

Performed basic data inspection including:

* Dataset shape
* Data types
* Missing values
* Duplicate records

---
**3. Feature Selection**

Selected numerical audio features relevant for clustering.

---
**4. Feature Scaling**

Standardized all features using StandardScaler to ensure equal contribution during clustering.

from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

---

**5. Optimal Cluster Selection**

Two evaluation techniques were used:

**Elbow Method**

Determines the optimal number of clusters by analyzing Within Cluster Sum of Squares (WCSS).

---

**Silhouette Score**

Measures clustering quality and separation between clusters.

---

**6. K-Means Clustering**

Applied K-Means clustering to segment songs into groups.

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)

---
**7. Cluster Visualization**

Principal Component Analysis (PCA) was used to reduce dimensionality and visualize clusters in two dimensions.

---
**8. Cluster Interpretation**

Cluster characteristics were analyzed using average feature values.


**Cluster 0 – Chill Acoustic Tracks**

Characteristics:

* Lower energy
* Higher acousticness
* Relaxed listening experience

from sklearn.decomposition import PCA

---
**Cluster 1 – Party Tracks**

Characteristics:

* High danceability
* High energy
* Faster tempo
* Positive mood (high valence)

---
**Cluster 2 – Rap / Live Performance Tracks**

Characteristics:

* High speechiness
* Higher liveness
* Moderate energy
* Strong rhythmic elements

---
**DBSCAN Clustering**

In addition to K-Means, the project also explores density-based clustering using DBSCAN.

from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.5, min_samples=5)

This helps identify:

* Dense song groups
* Outliers
* Non-spherical clusters

---
### Model Persistence

Trained models are saved using Joblib for future use.

joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

Saved files:

* kmeans_model.pkl
* scaler.pkl

---
### Output

The final clustered dataset is exported as:

final_clustered_songs.csv

This file contains cluster labels assigned to each song.

---
### Results

The clustering process successfully grouped songs based on similar musical attributes and listening patterns. The generated clusters can be used for:

* Automated playlist creation
* Personalized recommendations
* Music catalog organization
* Audio feature analysis
