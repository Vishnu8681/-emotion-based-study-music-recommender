import os
import pandas as pd
import re

# Resolve paths dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
train_path = os.path.join(project_root, 'datasets', 'train.txt')
output_path = os.path.join(project_root, 'datasets', 'cleaned_emotions.csv')

print("Loading data from:", train_path)
if not os.path.exists(train_path):
    # Fallback to current working directory
    train_path = 'train.txt'
    output_path = 'cleaned_emotions.csv'

df = pd.read_csv(train_path, sep=';', header=None, names=['text', 'emotion'])

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

df.to_csv(output_path, index=False)
print(f"\nSaved cleaned_emotions.csv to {output_path} — Phase 2 complete!")