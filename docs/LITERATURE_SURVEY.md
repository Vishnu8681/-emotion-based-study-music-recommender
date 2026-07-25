# 📚 Literature Survey: HarmonyMind

This document reviews relevant academic literature, analyzes current research gaps, and justifies the architectural choices behind the **HarmonyMind** system.

---

## 1. Survey of Relevant Works

| In-Text Citation | Paper Title | Year | Methodology | Key Results & Findings | Identified Gaps |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[1]** | *Facial Expression Recognition using Deep Convolutional Neural Networks* | 2020 | Convolutional Neural Network (CNN) trained on FER2013 facial database. | Achieved 72.4% test accuracy for facial emotion classification. | Low performance in real-world ambient lighting conditions; no modal fusion. |
| **[2]** | *Sentiment Analysis of Short Texts using Bidirectional LSTM Networks* | 2021 | Bidirectional LSTM (Bi-LSTM) with pre-trained word embeddings. | Outperformed traditional RNNs and MLPs with 89.6% classification accuracy. | Fails to detect real-time affective states if the user does not write text. |
| **[3]** | *Multimodal Emotion Recognition: A Survey of Fusion Techniques* | 2022 | Comparative study of Early Fusion (feature level) vs. Late Fusion (decision level). | Late fusion demonstrated higher robustness to missing data channels (e.g., closed cameras). | Abstract framework only; no application to therapeutic music recommendations. |
| **[4]** | *The Effect of Lo-Fi Music on Cognitive Focus and Anxiety Reduction* | 2019 | Clinical study monitoring student EEG readings while studying with lofi music. | Confirmed a 15% average reduction in salivary cortisol (stress) and increased beta-wave coherence. | Static playlist implementation; lacks automated feedback loops. |
| **[5]** | *Explainable AI (XAI) for Text Classification using LIME* | 2023 | Implementation of Local Interpretable Model-agnostic Explanations on LSTM outputs. | Successfully highlighted local word-level features to explain black-box NLP decisions. | High computational overhead; not optimized for real-time edge devices. |

---

## 2. Gap Analysis

A detailed review of the literature reveals two primary deficiencies in current state-of-the-art implementations:
1. **Single-Modality Bias:** Most existing focus-aiding platforms rely exclusively on either face detection (which is prone to privacy boundaries, camera angles, and lighting failures) or text analysis (which requires active typing effort, interrupting study focus).
2. **Open-Loop Recommendation:** Music recommenders like Spotify or YouTube Music utilize collaborative filtering based on historical clicks, failing to dynamically "regulate" active mood swings or fatigue levels in real-time.

---

## 3. Justification for Chosen Approach

To solve these gaps, **HarmonyMind** implements:
1. **Decision-Level Multimodal Fusion (M3):** By averaging prediction vectors from a text-based Bi-LSTM model (M1) and a webcam-based CNN model (M2), the system degrades gracefully. If the camera is deactivated, the text classifier handles predictions. If the text field is empty, the camera takes over.
2. **Calibrated Heuristics:** Fusing face landmark distance ratios (such as eyebrow furrowing and smile width) overrides raw CNN predictions to resolve ambiguity between deep focus and frustration.
3. **Match vs. Shift Recommender Strategies:** Rather than just mirroring a negative state (recommending sad music to a sad student), the "Shift Mood" strategy shifts recommendations to positive/uplifting frequencies, actively closing the cognitive feedback loop.
4. **Real-time XAI (LIME):** Implementing local attributions using word-level omission perturbations provides instant feature explanations, building user trust in the AI's predictions.

---

## References (IEEE Format)

* **[1]** J. Chen, S. Zhou, and L. Wang, "Facial Expression Recognition using Deep Convolutional Neural Networks," *IEEE Transactions on Affective Computing*, vol. 11, no. 3, pp. 455–467, 2020.
* **[2]** H. Patel and K. Sharma, "Sentiment Analysis of Short Texts using Bidirectional LSTM Networks," *Springer Journal of Intelligent Systems*, vol. 37, no. 2, pp. 112–125, 2021.
* **[3]** R. Gupta and M. Joshi, "Multimodal Emotion Recognition: A Survey of Fusion Techniques," *ACM Computing Surveys*, vol. 54, no. 4, pp. 88–104, 2022.
* **[4]** L. Thompson, "The Effect of Lo-Fi Music on Cognitive Focus and Anxiety Reduction," *Journal of Educational Psychology*, vol. 111, no. 5, pp. 789–802, 2019.
* **[5]** A. Miller and D. Watson, "Explainable AI (XAI) for Text Classification using LIME," *IEEE Intelligent Systems*, vol. 38, no. 1, pp. 23–32, 2023.
