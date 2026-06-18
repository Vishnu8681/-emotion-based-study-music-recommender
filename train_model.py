import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

print("Step 1: Loading cleaned data...")
df = pd.read_csv('cleaned_emotions.csv')
print("Loaded:", df.shape)

print("\nStep 2: Preparing text and labels...")
MAX_WORDS = 10000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df['text'])
X = pad_sequences(tokenizer.texts_to_sequences(df['text']), maxlen=MAX_LEN)

le = LabelEncoder()
y = le.fit_transform(df['emotion'])
print("Emotions found:", list(le.classes_))

print("\nStep 3: Building the model...")
model = Sequential([
    Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

print("\nStep 4: Training the model (this takes 2-5 minutes)...")
history = model.fit(
    X, y,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

print("\nStep 5: Saving the model and tokenizer...")
model.save('emotion_model.h5')

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("\nStep 6: Testing with a sample sentence...")
def predict_emotion(text):
    seq = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=MAX_LEN)
    pred = model.predict(seq, verbose=0)
    return le.classes_[np.argmax(pred)]

test_sentences = [
    "I am so happy today!",
    "I feel really sad and lonely",
    "This makes me so angry",
    "I am scared and nervous",
    "I love this so much"
]

print("\nSample predictions:")
for sentence in test_sentences:
    print(f"  '{sentence}' → {predict_emotion(sentence)}")

print("\nPhase 3 complete! Model saved as emotion_model.h5")