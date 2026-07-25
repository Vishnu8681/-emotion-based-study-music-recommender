# 🌌 HarmonyMind – Emotion-Based Study Music Recommender using Deep Learning

**Course Code:** 23ADC04 DEEP LEARNING (Individual Project)  
**Student Name:** Vishnu Priya S  
**Roll Number:** 23ADC04  
**Section / Year:** B.Tech AI & DS / III Year  
**Institution:** Sri Krishna College of Engineering and Technology  

---

## 🎯 Project Objective & Abstract

HarmonyMind is an intelligent, real-time closed-loop study companion designed to optimize student focus and alleviate self-study anxiety. Traditional music recommenders do not monitor affective states in real-time. HarmonyMind resolves this by utilizing multimodal Deep Learning architectures to process student text diary entries and webcam expression frames. It averages predictions at a late-stage decision fusion layer, resolving state ambiguities through client-side MediaPipe landmark metrics. Recommended songs are routed using a "Vibe Strategy" (Match Mood vs. Shift Mood) to regulate focus or stress levels.

---

## ⚙️ Repository Folder Structure

The project has been organized according to professional academic standards:

```
harmony_mind/
├── src/                      # Core backend source scripts
│   ├── app.py                # Flask Server (with LIME XAI, PDF report, CSV export)
│   ├── prepare_data.py       # Preprocessing and text cleaning pipeline
│   ├── train_model.py        # Keras Bi-LSTM training loop (with early stopping, callbacks)
│   └── generate_final_report.py # ReportLab PDF report compiler
├── models/                   # Serialized neural networks
│   ├── best_model.h5         # Saved checkpoint weights (EarlyStopped best epoch)
│   ├── emotion_model.keras   # Backend Flask model weights
│   ├── tokenizer.pkl         # Trained text tokenizer
│   └── label_encoder.pkl     # Encoded target emotions
├── datasets/                 # Datasets directory
│   ├── train.txt             # 16,000 sentence training dataset
│   ├── val.txt               # 2,000 sentence validation dataset
│   ├── test.txt              # 2,000 sentence test dataset
│   ├── cleaned_emotions.csv  # Output preprocessed CSV
│   └── music.csv             # Song catalog database (artist, YouTube link)
├── evaluation/               # Output performance charts
│   ├── training_curves.png   # Accuracy & Loss curves
│   ├── confusion_matrix.png  # Multiclass confusion matrix
│   ├── roc_curves.png        # One-vs-Rest ROC curve outputs
│   └── classification_report.txt # Accuracy, Precision, Recall, F1 scores
├── notebooks/                # Jupyter Notebooks
│   ├── EDA.ipynb             # Exploratory Data Analysis
│   ├── Training.ipynb        # Model Training loop
│   ├── Evaluation.ipynb      # Model Evaluation & naive bayes baseline comparison
│   ├── Inference.ipynb       # Sample classifier & explainable LIME attributions
│   └── Visualization.ipynb   # Seaborn & Matplotlib evaluations
├── docs/                     # Academic Documentation
│   ├── PROPOSAL.md           # Project proposal
│   ├── LITERATURE_SURVEY.md  # Detailed literature review & gap analysis
│   ├── FINAL_REPORT.pdf      # IEEE format compiled PDF report
│   └── architecture/
│       └── architecture.png  # Programmatic architecture block diagram
├── templates/                # Frontend views
│   └── index.html            # Premium Glassmorphic AI UI dashboard
├── requirements.txt          # Python packages list
├── .gitignore                # Git exclusions
├── LICENSE                   # Open-source license
└── README.md                 # Project landing page (this document)
```

---

## 📈 Evaluation Results & Comparison

HarmonyMind compares the performance of its Bidirectional LSTM text classifier against a standard **TF-IDF + Naive Bayes** baseline:

| Metric | TF-IDF + Naive Bayes Baseline | Bidirectional LSTM (HarmonyMind) |
| :--- | :--- | :--- |
| **Test Accuracy** | 67.20% | **90.85%** |
| **Weighted F1-Score** | 64.80% | **91.00%** |
| **Val Loss** | N/A | **0.2494** |
| **Early Stopping** | No | **Yes (Epoch 5 restored)** |

---

## 🚀 How to Run the Project

### Prerequisites
Make sure Python 3.10+ and virtualenv are installed.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/username/DL_HarmonyMind_23ADC04.git
   cd DL_HarmonyMind_23ADC04
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Preprocessing and Training (Optional):**
   *To rebuild the datasets and re-train the model weights:*
   ```bash
   python src/prepare_data.py
   python src/train_model.py
   ```

4. **Launch the Web Application:**
   ```bash
   python src/app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000` to interact with the glassmorphic AI dashboard.
