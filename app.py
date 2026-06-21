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

def predict_text_emotion(text):
    seq = pad_sequences(
        tokenizer.texts_to_sequences([text]),
        maxlen=MAX_LEN
    )
    pred = model.predict(seq, verbose=0)
    emotion = le.classes_[np.argmax(pred)]
    confidence = round(float(np.max(pred)) * 100, 1)
    return emotion, confidence

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
    emotion, confidence = predict_text_emotion(text)
    songs = get_songs(emotion)
    return jsonify({
        'emotion': emotion,
        'confidence': confidence,
        'recommendations': songs,
        'method': 'text'
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

        raw_emotion = result[0]['dominant_emotion']
        emotion = EMOTION_MAP.get(raw_emotion, 'neutral')
        confidence = round(float(result[0]['emotion'][raw_emotion]), 1)

        songs = get_songs(emotion)
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'recommendations': songs,
            'method': 'webcam',
            'raw_emotion': raw_emotion
        })

    except Exception as e:
        import traceback
        print("WEBCAM ERROR:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)