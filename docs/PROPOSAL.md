# 📝 Project Proposal: HarmonyMind

**Title:** HarmonyMind – Emotion-Based Study Music Recommender using Deep Learning  
**Course Code:** 23ADC04 DEEP LEARNING (Individual Project)  

---

## 1. Objective & Problem Statement

### Problem Statement
Modern students face high levels of stress, cognitive fatigue, and distractions during self-study sessions. While background music (specifically Lo-Fi study beats and ambient noise) is scientifically proven to enhance focus and regulate anxiety, static playlists do not adapt to a student's changing emotional states. 

If a student is feeling frustrated (e.g., due to code bugs) or exhausted, they require soothing ambient sounds or relaxing frequencies. If they are in a positive flow state, they require uplifting tracks to sustain momentum. Existing music services rely on manual search or generic algorithmic play queues that lack real-time physiological/affective awareness.

### Expected Outcome
**HarmonyMind** addresses this gap by building an intelligent, real-time closed-loop music recommendation application. By utilizing Deep Learning architectures across two modalities—natural language (text diary inputs) and facial micro-expressions (webcam feeds)—the system calculates an immediate emotion confidence vector. It then recommends tailored lofi music tracks using a double-action "Vibe Strategy" (Match Mood vs. Shift Mood) to optimize cognitive performance.

---

## 2. Dataset Description

The system utilizes two primary datasets:
1. **CARER Emotion Dataset (Text):** 
   - A collection of over 20,000 text statements mapped to six basic emotion classes: `joy`, `sadness`, `anger`, `fear`, `surprise`, and `love`.
   - The data is split into `train.txt` (16,000 samples), `val.txt` (2,000 samples), and `test.txt` (2,000 samples).
   - Source: Kaggle / Twitter Emotion Dataset.
2. **HarmonyMind Music Database (`music.csv`):**
   - A curated dataset of 34 high-quality study tracks categorised by emotional compatibility (`happy`, `sad`, `angry`, `fear`, `neutral`, `surprise`, `disgust`, `joy`, `sadness`, `anger`, `love`).
   - Contains direct YouTube streaming URL endpoints and artist attribution metadata.

---

## 3. Proposed Architecture & Methodology

HarmonyMind employs a hybrid, multi-modal decision-level fusion model mapping three modules:

```
[User Text Input] ---> [Bi-LSTM Text Encoder (M1)] ---\
                                                       +---> [Multimodal Fusion Decision (M3)] ---> [Music Recommendation]
[User Face Video] ---> [DeepFace SSD/CNN Model (M2)]  -/
```

* **Module 1 (M1) - Text Sentiment Encoder:** A Bidirectional LSTM network with an embedding layer that processes student diary entries to identify the underlying emotional state.
* **Module 2 (M2) - Facial Emotion Recognizer:** A Deep Convolutional Neural Network (CNN) using the SSD face detector backend (via DeepFace) that processes real-time webcam frame grabs to identify facial landmarks and micro-expressions.
* **Module 3 (M3) - Multimodal Fusion & Recommendation Engine:**
  - Standardizes the probability scores from M1 and M2.
  - Combines them using decision-level fusion (probability averaging).
  - Integrates direct client-side landmark heuristics (e.g., mouth shape, eyebrow furrowing) to resolve neutral-state ambiguities.
  - Filters tracks using "Match" (sustain state) or "Shift" (therapeutic mood regulation) logic.
