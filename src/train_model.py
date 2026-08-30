import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing.text import Tokenizer
# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Configure matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
datasets_dir = os.path.join(project_root, 'datasets')
models_dir = os.path.join(project_root, 'models')
eval_dir = os.path.join(project_root, 'evaluation')

os.makedirs(models_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

train_path = os.path.join(datasets_dir, 'train.txt')
val_path = os.path.join(datasets_dir, 'val.txt')
test_path = os.path.join(datasets_dir, 'test.txt')

print("Step 1: Loading train, val, and test data...")
def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing dataset file: {filepath}")
    df = pd.read_csv(filepath, sep=';', header=None, names=['text', 'emotion'])
    return df

df_train = load_data(train_path)
df_val = load_data(val_path)
df_test = load_data(test_path)

print(f"Loaded train shape: {df_train.shape}, val: {df_val.shape}, test: {df_test.shape}")

# Preprocessing function
import re
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = text.strip()
    return text

print("Cleaning texts...")
df_train['clean_text'] = df_train['text'].apply(clean_text)
df_val['clean_text'] = df_val['text'].apply(clean_text)
df_test['clean_text'] = df_test['text'].apply(clean_text)

# Set params
MAX_WORDS = 10000
MAX_LEN = 100

print("\nStep 2: Tokenizing and padding sequences...")
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df_train['clean_text'])

X_train = pad_sequences(tokenizer.texts_to_sequences(df_train['clean_text']), maxlen=MAX_LEN)
X_val = pad_sequences(tokenizer.texts_to_sequences(df_val['clean_text']), maxlen=MAX_LEN)
X_test = pad_sequences(tokenizer.texts_to_sequences(df_test['clean_text']), maxlen=MAX_LEN)

le = LabelEncoder()
y_train = le.fit_transform(df_train['emotion'])
y_val = le.transform(df_val['emotion'])
y_test = le.transform(df_test['emotion'])

num_classes = len(le.classes_)
print("Emotions found:", list(le.classes_))

# Save tokenizers & encoders
with open(os.path.join(models_dir, 'tokenizer.pkl'), 'wb') as f:
    pickle.dump(tokenizer, f)
with open(os.path.join(models_dir, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)

print("\nStep 3: Building the Deep Learning Model (LSTM)...")
model = Sequential([
    # pyrefly: ignore [unexpected-keyword]
    Embedding(MAX_WORDS, 64, input_shape=(MAX_LEN,)),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Define checkpoints and callbacks
best_model_path = os.path.join(models_dir, 'best_model.h5')
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=best_model_path, monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001, verbose=1)
]

print("\nStep 4: Training the LSTM model with callbacks...")
# Train for 5 epochs to run quickly while achieving high accuracy
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# Load best weights
if os.path.exists(best_model_path):
    print("\nLoaded best checkpoint weights.")
    model = tf.keras.models.load_model(best_model_path)
else:
    print("\nSaving final weights as best_model.h5 (no checkpoint found).")
    model.save(best_model_path)

# Save keras model as emotion_model.keras as well for Flask backend compatibility
flask_model_path = os.path.join(models_dir, 'emotion_model.keras')
model.save(flask_model_path)
print(f"Model saved in Keras format at {flask_model_path}")

print("\nStep 5: Training a Baseline Model (TF-IDF + Naive Bayes) for Comparison...")
tfidf = TfidfVectorizer(max_features=MAX_WORDS)
X_train_tfidf = tfidf.fit_transform(df_train['clean_text'])
X_test_tfidf = tfidf.transform(df_test['clean_text'])

baseline_model = MultinomialNB()
baseline_model.fit(X_train_tfidf, y_train)
baseline_acc = baseline_model.score(X_test_tfidf, y_test)
print(f"Baseline (Naive Bayes) Accuracy: {baseline_acc * 100:.2f}%")

print("\nStep 6: Evaluating Model and Generating Reports...")
# Predict on test
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Calculate metrics
lstm_precision, lstm_recall, lstm_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
lstm_acc = np.mean(y_test == y_pred)

print(f"\nLSTM Test Accuracy: {lstm_acc * 100:.2f}%")

# Save Classification Report
clf_report = classification_report(y_test, y_pred, target_names=le.classes_)
print("\nClassification Report:")
print(clf_report)

report_text_path = os.path.join(eval_dir, 'classification_report.txt')
with open(report_text_path, 'w') as f:
    f.write("=== LSTM EMOTION CLASSIFIER EVALUATION REPORT ===\n\n")
    f.write(f"Test Accuracy: {lstm_acc * 100:.2f}%\n")
    f.write(f"Weighted Precision: {lstm_precision * 100:.2f}%\n")
    f.write(f"Weighted Recall: {lstm_recall * 100:.2f}%\n")
    f.write(f"Weighted F1-Score: {lstm_f1 * 100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(clf_report)
    f.write("\n\n=== BASELINE MODEL COMPARISON ===\n")
    f.write(f"Baseline Naive Bayes Accuracy: {baseline_acc * 100:.2f}%\n")
    f.write(f"LSTM Improvement over Baseline: {(lstm_acc - baseline_acc) * 100:.2f}%\n")
print(f"Saved classification report to {report_text_path}")

# Plot and save curves
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='#6366f1', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='#10b981', marker='s')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='#ef4444', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', color='#f59e0b', marker='s')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
curves_path = os.path.join(eval_dir, 'training_curves.png')
plt.savefig(curves_path, dpi=300)
plt.close()
print(f"Saved training curves to {curves_path}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix on Test Dataset')
plt.ylabel('True Emotion')
plt.xlabel('Predicted Emotion')
plt.tight_layout()
cm_path = os.path.join(eval_dir, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"Saved confusion matrix to {cm_path}")

# Multi-class ROC curve
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test == i, y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{le.classes_[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Multi-class')
plt.legend(loc="lower right")
plt.tight_layout()
roc_path = os.path.join(eval_dir, 'roc_curves.png')
plt.savefig(roc_path, dpi=300)
plt.close()
print(f"Saved ROC curves to {roc_path}")

print("\nModel Training and Evaluation complete! Saved best model weights to models/best_model.h5.")
