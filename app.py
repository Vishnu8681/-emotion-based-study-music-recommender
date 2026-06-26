import random
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
import base64
import cv2
from deepface import DeepFace
import keras
from keras.preprocessing.sequence import pad_sequences
app = Flask(__name__)

print("Loading model...")
model = keras.models.load_model('emotion_model.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

music_df = pd.read_csv('music.csv')
print("All loaded! Flask is ready.")

MAX_LEN = 100

EMOTION_MAP = {
    'happy': 'joy',
    'sad': 'sadness',
    'angry': 'anger',
    'fear': 'fear',
    'surprise': 'surprise',
    'disgust': 'disgust',
    'neutral': 'neutral'
}
QUOTES = {
    'joy': [
        "Happiness is not by chance, but by choice.",
        "The best way to spread positivity is to enjoy your own.",
        "Keep this energy — it's contagious."
    ],
    'sadness': [
        "Even the darkest night will end and the sun will rise.",
        "Tough times don't last, tough people do.",
        "It's okay to rest. You're allowed to feel this."
    ],
    'anger': [
        "Calm mind brings inner strength and self-confidence.",
        "Breathe. This feeling will pass.",
        "Channel this energy into focus, not frustration."
    ],
    'fear': [
        "Courage is not the absence of fear, but moving through it.",
        "You have survived 100% of your hardest days so far.",
        "One step at a time is still progress."
    ],
    'love': [
        "Where there is love, there is life.",
        "Carry this warmth into everything you do today.",
        "Gratitude turns what we have into enough."
    ],
    'surprise': [
        "Stay curious — the best ideas come from surprise.",
        "Embrace the unexpected, it often leads somewhere good.",
        "Every surprise is a new door opening."
    ],
    'neutral': [
        "Calm is a superpower.",
        "Steady minds make the clearest decisions.",
        "Focus is a quiet kind of strength."
    ],
    'disgust': [
        "Let go of what doesn't serve you.",
        "Clarity often comes right after discomfort.",
        "This feeling is temporary — your focus isn't."
    ]
}

def predict_text_emotion(text):
    seq = pad_sequences(
        tokenizer.texts_to_sequences([text]),
        maxlen=MAX_LEN
    )
    pred = model.predict(seq, verbose=0)[0]
    emotion = le.classes_[np.argmax(pred)]
    confidence = round(float(np.max(pred)) * 100, 1)
    breakdown = {le.classes_[i]: round(float(pred[i]) * 100, 1) for i in range(len(le.classes_))}
    return emotion, confidence, breakdown

def get_songs(emotion):
    songs = music_df[music_df['emotion'] == emotion]
    if len(songs) == 0:
        songs = music_df.sample(3)
    else:
        songs = songs.sample(min(3, len(songs)))
    return songs[['song', 'artist', 'youtube_link']].to_dict('records')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend/text', methods=['POST'])
def recommend_text():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    emotion, confidence, breakdown = predict_text_emotion(text)
    songs = get_songs(emotion)
    quote = random.choice(QUOTES.get(emotion, QUOTES['neutral']))
    return jsonify({
        'emotion': emotion,
        'confidence': confidence,
        'recommendations': songs,
        'method': 'text',
        'quote': quote,
        'breakdown': breakdown
    })

@app.route('/recommend/webcam', methods=['POST'])
def recommend_webcam():
    try:
        data = request.get_json()
        image_data = data.get('image', '')

        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
)

        emotions_dict = result[0]['emotion']
        if 'neutral' in emotions_dict:
            emotions_dict['neutral'] = emotions_dict['neutral'] * 0.6
        raw_emotion = max(emotions_dict, key=emotions_dict.get)
        emotion = EMOTION_MAP.get(raw_emotion, 'neutral')
        confidence = round(float(emotions_dict[raw_emotion]), 1)
        breakdown = {EMOTION_MAP.get(k, k): round(float(v), 1) for k, v in emotions_dict.items()}

        songs = get_songs(emotion)
        quote = random.choice(QUOTES.get(emotion, QUOTES['neutral']))
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'recommendations': songs,
            'method': 'webcam',
            'raw_emotion': raw_emotion,
            'quote': quote,
            'breakdown': breakdown
        })

    except Exception as e:
        import traceback
        print("WEBCAM ERROR:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)