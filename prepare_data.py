import pandas as pd
import re

print("Loading data...")
df = pd.read_csv('train.txt', sep=';', header=None, names=['text', 'emotion'])

print("Shape:", df.shape)
print("\nEmotion counts:")
print(df['emotion'].value_counts())
print("\nFirst 5 rows:")
print(df.head())

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = text.strip()
    return text

print("\nCleaning text...")
df['text'] = df['text'].apply(clean_text)

df.to_csv('cleaned_emotions.csv', index=False)
print("\nSaved cleaned_emotions.csv — Phase 2 complete!")