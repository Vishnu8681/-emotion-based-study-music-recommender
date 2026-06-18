from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

print("Loading model...")
model = load_model('emotion_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

music_df = pd.read_csv('music.csv')
print("All loaded! Flask is ready.")

MAX_LEN = 100

def predict_emotion(text):
    seq = pad_sequences(
        tokenizer.texts_to_sequences([text]),
        maxlen=MAX_LEN
    )
    pred = model.predict(seq, verbose=0)
    emotion = le.classes_[np.argmax(pred)]
    confidence = round(float(np.max(pred)) * 100, 1)
    return emotion, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    emotion, confidence = predict_emotion(text)

    songs = music_df[music_df['emotion'] == emotion]
    if len(songs) == 0:
        songs = music_df.sample(3)
    else:
        songs = songs.sample(min(3, len(songs)))

    recommendations = songs[['song', 'artist', 'youtube_link']].to_dict('records')

    return jsonify({
        'emotion': emotion,
        'confidence': confidence,
        'recommendations': recommendations
    })

if __name__ == '__main__':
    app.run(debug=True)