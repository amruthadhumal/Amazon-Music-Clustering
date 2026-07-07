import streamlit as st
import pandas as pd
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans_model.pkl')
df = pd.read_csv('single_genre_artists.csv')
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

st.title("Amazon Music Song Cluster Predictor")

st.write("Enter song audio features to predict cluster:")

inputs = {}
for f in features:
    inputs[f] = st.slider(f,
    min_value=0.0 if f in ['danceability','energy','speechiness','acousticness','instrumentalness','liveness','valence'] else -30.0 if f=='loudness' else 0.0,
    max_value=1.0 if f in ['danceability','energy','speechiness','acousticness','instrumentalness','liveness','valence'] else -0.0 if f=='loudness' else 200.0 if f=='tempo' else 200000.0,
    value=0.5 if f in ['danceability','energy','speechiness','acousticness','instrumentalness','liveness','valence'] else -5.0 if f=='loudness' else 120.0 if f=='tempo' else 180000.0,
    step=0.01 if f!='duration_ms' else 1000.0)

input_df = pd.DataFrame([inputs])
input_scaled = scaler.transform(input_df)
cluster = kmeans.predict(input_scaled)[0]

st.write(f"Predicted Cluster: {cluster}")

if cluster == 0:
    st.write("Energetic Dance Tracks")
elif cluster == 1:
    st.write("Balanced Vocal Songs")
else:
    st.write("Calm Acoustic")

# PCA viz
X_scaled = scaler.transform(df[features])
pca = PCA(2)
X_pca = pca.fit_transform(X_scaled)
fig, ax = plt.subplots()
ax.scatter(X_pca[:,0], X_pca[:,1], c=kmeans.labels_, cmap='viridis', alpha=0.1)
ax.scatter(pca.transform(input_scaled)[:,0], pca.transform(input_scaled)[:,1], c='red', s=100)
st.pyplot(fig)