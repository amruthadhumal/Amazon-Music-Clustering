import streamlit as st
import pandas as pd
import joblib

# Load Model and Scaler
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Amazon Music Clustering",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Amazon Music Clustering")
st.write(
    "Group songs based on audio features using K-Means Clustering."
)

st.sidebar.header("Song Features")

danceability = st.sidebar.slider(
    "Danceability", 0.0, 1.0, 0.50
)

energy = st.sidebar.slider(
    "Energy", 0.0, 1.0, 0.50
)

loudness = st.sidebar.slider(
    "Loudness", -60.0, 5.0, -10.0
)

speechiness = st.sidebar.slider(
    "Speechiness", 0.0, 1.0, 0.10
)

acousticness = st.sidebar.slider(
    "Acousticness", 0.0, 1.0, 0.50
)

instrumentalness = st.sidebar.slider(
    "Instrumentalness", 0.0, 1.0, 0.00
)

liveness = st.sidebar.slider(
    "Liveness", 0.0, 1.0, 0.20
)

valence = st.sidebar.slider(
    "Valence", 0.0, 1.0, 0.50
)

tempo = st.sidebar.slider(
    "Tempo", 40.0, 250.0, 120.0
)

duration_ms = st.sidebar.number_input(
    "Duration (milliseconds)",
    min_value=10000,
    value=210000
)

if st.sidebar.button("Find Cluster"):

    song_data = pd.DataFrame({
        "danceability":[danceability],
        "energy":[energy],
        "loudness":[loudness],
        "speechiness":[speechiness],
        "acousticness":[acousticness],
        "instrumentalness":[instrumentalness],
        "liveness":[liveness],
        "valence":[valence],
        "tempo":[tempo],
        "duration_ms":[duration_ms]
    })

    scaled_data = scaler.transform(song_data)

    cluster = kmeans.predict(scaled_data)[0]

    st.success(
        f"This song belongs to Cluster {cluster}"
    )

    if cluster == 0:
        st.info(
            "Cluster 0: Mostly acoustic and relaxing songs."
        )

    elif cluster == 1:
        st.info(
            "Cluster 1: High-energy and dance-friendly songs."
        )

    elif cluster == 2:
        st.info(
            "Cluster 2: Speech-focused or live-performance songs."
        )

st.subheader("Feature Summary")

st.dataframe(
    pd.DataFrame({
        "Feature":[
            "Danceability",
            "Energy",
            "Loudness",
            "Speechiness",
            "Acousticness",
            "Instrumentalness",
            "Liveness",
            "Valence",
            "Tempo",
            "Duration"
        ],
        "Value":[
            danceability,
            energy,
            loudness,
            speechiness,
            acousticness,
            instrumentalness,
            liveness,
            valence,
            tempo,
            duration_ms
        ]
    })
)

st.markdown("---")
st.markdown(
    "Built using K-Means Clustering, Scikit-Learn and Streamlit."
)
