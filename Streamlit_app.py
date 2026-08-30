import streamlit as st
import pandas as pd
import pickle
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="HarmonyMind", page_icon="🎵")

model = load_model("emotion_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

music_df = pd.read_csv("music.csv")

st.title("🎵 HarmonyMind")
st.write("Emotion-Based Study Music Recommender")

text = st.text_area("How are you feeling today?")

if st.button("Recommend Music"):
    if text.strip():
        seq = tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=50)
        pred = model.predict(pad, verbose=0)
        emotion = label_encoder.inverse_transform([pred.argmax()])[0]

        st.success(f"Detected Emotion: {emotion}")

        songs = music_df[music_df["Emotion"].str.lower() == emotion.lower()]

        if not songs.empty:
            song = songs.sample(1).iloc[0]
            st.write(f"**Song:** {song['Song']}")
            st.write(f"**Artist:** {song['Artist']}")
            st.write(song["Link"])
        else:
            st.info("No song found for this emotion.")
    else:
        st.warning("Enter some text.")