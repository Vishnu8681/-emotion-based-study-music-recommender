import random
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
import base64
import cv2
import sys
from deepface import DeepFace
import keras
from keras.preprocessing.sequence import pad_sequences

# Configure stdout and stderr to use UTF-8 encoding on Windows to prevent DeepFace emoji print crashes
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import os

# Resolve directory paths dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

app = Flask(
    __name__,
    template_folder=os.path.join(project_root, 'templates'),
    static_folder=os.path.join(project_root, 'static')
)

print("Loading model from restructured directories...")
model_path = os.path.join(project_root, 'models', 'emotion_model.keras')
tokenizer_path = os.path.join(project_root, 'models', 'tokenizer.pkl')
label_encoder_path = os.path.join(project_root, 'models', 'label_encoder.pkl')
music_csv_path = os.path.join(project_root, 'datasets', 'music.csv')

# Fallbacks in case directory resolution is run from root in some envs
if not os.path.exists(model_path):
    model_path = 'emotion_model.keras'
    tokenizer_path = 'tokenizer.pkl'
    label_encoder_path = 'label_encoder.pkl'
    music_csv_path = 'music.csv'

model = keras.models.load_model(model_path)
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

with open(label_encoder_path, 'rb') as f:
    le = pickle.load(f)

music_df = pd.read_csv(music_csv_path)
print("All loaded! Flask is ready from path:", project_root)


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

EMOTION_GROUPS = {
    'joy': ['happy', 'joy', 'energetic'],
    'happy': ['happy', 'joy', 'energetic'],
    'love': ['happy', 'joy'],
    'sadness': ['sad', 'sadness'],
    'sad': ['sad', 'sadness'],
    'anger': ['angry'],
    'angry': ['angry'],
    'fear': ['fear'],
    'surprise': ['surprise'],
    'disgust': ['disgust'],
    'neutral': ['neutral']
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

def explain_text_lime(text, target_emotion):
    words = text.split()
    if len(words) == 0:
        return []
    
    # Get baseline sequence prediction
    seq = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=MAX_LEN)
    pred = model.predict(seq, verbose=0)[0]
    
    try:
        class_idx = list(le.classes_).index(target_emotion)
    except ValueError:
        class_idx = np.argmax(pred)
        
    orig_conf = float(pred[class_idx])
    
    attributions = []
    for i in range(len(words)):
        # Remove the i-th word
        perturbed_words = words[:i] + words[i+1:]
        perturbed_text = " ".join(perturbed_words)
        
        if len(perturbed_words) == 0:
            perturbed_conf = 0.0
        else:
            pert_seq = pad_sequences(tokenizer.texts_to_sequences([perturbed_text]), maxlen=MAX_LEN)
            pert_pred = model.predict(pert_seq, verbose=0)[0]
            perturbed_conf = float(pert_pred[class_idx])
            
        score = orig_conf - perturbed_conf
        attributions.append({
            "word": words[i],
            "score": round(score * 100, 2)
        })
        
    return attributions

def get_songs(emotion, strategy='match'):
    target_emotion = emotion
    if strategy == 'shift':
        if emotion in ['sadness', 'sad']:
            target_emotion = 'joy'
        elif emotion in ['anger', 'angry', 'fear', 'disgust']:
            target_emotion = 'neutral'
            
    allowed_emotions = EMOTION_GROUPS.get(target_emotion, [target_emotion])
    songs = music_df[music_df['emotion'].isin(allowed_emotions)]
    if len(songs) == 0:
        songs = music_df.sample(4)
    else:
        songs = songs.sample(min(4, len(songs)))
    return songs[['song', 'artist', 'youtube_link']].to_dict('records')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend/text', methods=['POST'])
def recommend_text():
    data = request.get_json()
    text = data.get('text', '')
    strategy = data.get('strategy', 'match')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    emotion, confidence, breakdown = predict_text_emotion(text)
    lime_attributions = explain_text_lime(text, emotion)
    songs = get_songs(emotion, strategy)
    quote = random.choice(QUOTES.get(emotion, QUOTES['neutral']))
    return jsonify({
        'emotion': emotion,
        'confidence': confidence,
        'recommendations': songs,
        'method': 'text',
        'quote': quote,
        'breakdown': breakdown,
        'lime_attributions': lime_attributions
    })

def apply_client_metrics(emotions_dict, client_metrics):
    if not client_metrics:
        return emotions_dict
    
    smile_score = client_metrics.get('smileScore', 0.0)
    furrow_score = client_metrics.get('eyebrowFurrow', 1.0)
    surprise_score = client_metrics.get('surpriseScore', 0.0)
    
    # Custom rule overrides from face landmarks
    is_surprise_overridden = client_metrics.get('isSurpriseOverridden', False)
    is_sad_hand_on_cheek = client_metrics.get('isSadHandOnCheek', False)
    is_sad_lips_down = client_metrics.get('isSadLipsDown', False)
    is_angry_squint = client_metrics.get('isAngrySquint', False)
    is_asymmetric_angry = client_metrics.get('isAsymmetricAngry', False)

    if is_surprise_overridden:
        emotions_dict['surprise'] = 1000.0
        if 'neutral' in emotions_dict:
            emotions_dict['neutral'] = 0.0
    elif is_sad_hand_on_cheek:
        emotions_dict['sad'] = 1000.0
        if 'neutral' in emotions_dict:
            emotions_dict['neutral'] = 0.0
    elif is_sad_lips_down:
        emotions_dict['sad'] = 1000.0
        if 'neutral' in emotions_dict:
            emotions_dict['neutral'] = 0.0
    elif is_angry_squint:
        emotions_dict['angry'] = 1000.0
        if 'neutral' in emotions_dict:
            emotions_dict['neutral'] = 0.0
    elif is_asymmetric_angry:
        emotions_dict['angry'] = 1000.0
        if 'neutral' in emotions_dict:
            emotions_dict['neutral'] = 0.0
    else:
        # 1. Smile Calibration: Boost happy/joy
        if smile_score > 0.35:
            emotions_dict['happy'] = emotions_dict.get('happy', 0.0) + smile_score * 120.0
            if 'neutral' in emotions_dict:
                emotions_dict['neutral'] *= 0.1
                
        # 2. Eyebrow Furrow Calibration: Boost angry, sad (concentration/frown states)
        if furrow_score < 0.65:
            intensity = max(0.0, (0.65 - furrow_score) / 0.65)
            emotions_dict['angry'] = emotions_dict.get('angry', 0.0) + intensity * 60.0
            emotions_dict['sad'] = emotions_dict.get('sad', 0.0) + intensity * 40.0
            if 'neutral' in emotions_dict:
                emotions_dict['neutral'] *= 0.5
                
        # 3. Surprise Calibration: Boost surprise
        if surprise_score > 0.45:
            emotions_dict['surprise'] = emotions_dict.get('surprise', 0.0) + surprise_score * 100.0
            if 'neutral' in emotions_dict:
                emotions_dict['neutral'] *= 0.2
            
    return emotions_dict

@app.route('/recommend/webcam', methods=['POST'])
def recommend_webcam():
    try:
        data = request.get_json()
        image_data = data.get('image', '')
        strategy = data.get('strategy', 'match')
        client_metrics = data.get('clientMetrics', {})

        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='ssd'
        )

        emotions_dict = result[0]['emotion']
        if 'neutral' in emotions_dict:
            emotions_dict['neutral'] = emotions_dict['neutral'] * 0.25

        # Fuse client landmarks metrics to increase capture accuracy
        emotions_dict = apply_client_metrics(emotions_dict, client_metrics)

        raw_emotion = max(emotions_dict, key=emotions_dict.get)
        emotion = EMOTION_MAP.get(raw_emotion, 'neutral')
        
        # Calculate normalized confidence (cast to float to avoid numpy serialization errors)
        total_sum = sum(emotions_dict.values())
        if total_sum > 0:
            confidence = min(100.0, float(round((emotions_dict[raw_emotion] / total_sum) * 100.0, 1)))
        else:
            confidence = float(round(float(emotions_dict[raw_emotion]), 1))

        breakdown = {EMOTION_MAP.get(k, k): float(round(float(v), 1)) for k, v in emotions_dict.items()}
        bd_sum = sum(breakdown.values())
        if bd_sum > 0:
            breakdown = {k: float(round((v / bd_sum) * 100.0, 1)) for k, v in breakdown.items()}

        # Override with exact songs suitable for custom sad reactions if strategy is 'match'
        is_sad_hand_on_cheek = client_metrics.get('isSadHandOnCheek', False)
        is_sad_lips_down = client_metrics.get('isSadLipsDown', False)
        if strategy == 'match' and is_sad_hand_on_cheek:
            songs = [
                {"song": "Po Nee Po (Tamil Lofi Flip)", "artist": "Anirudh Ravichander (Lofi)", "youtube_link": "https://www.youtube.com/watch?v=FPpaBN_Go_I"},
                {"song": "The Night We Met (English Lofi)", "artist": "Lord Huron (Lofi)", "youtube_link": "https://www.youtube.com/watch?v=WxA-agYHG4w"},
                {"song": "Kanave Kanave (Tamil Lofi Flip)", "artist": "Anirudh Ravichander (Lofi)", "youtube_link": "https://www.youtube.com/watch?v=7Bx6S1I9t1s"},
                {"song": "Fix You (English Lofi)", "artist": "Coldplay (Lofi)", "youtube_link": "https://www.youtube.com/watch?v=T42VNg6KhWo"}
            ]
        elif strategy == 'match' and is_sad_lips_down:
            songs = [
                {"song": "Kannaana Kanney (Tamil Lofi)", "artist": "D. Imman (Lofi)", "youtube_link": "https://www.youtube.com/watch?v=C0KquTdmuoM"},
                {"song": "Someone Like You (English Lofi)", "artist": "Adele (Lofi)", "youtube_link": "https://www.youtube.com/watch?v=grT_dzV0QUQ"},
                {"song": "Maruvaarthai (Tamil Lofi)", "artist": "Sid Sriram (Lofi)", "youtube_link": "https://www.youtube.com/watch?v=9uZpMnXKLq8"},
                {"song": "Hurt (English Lofi)", "artist": "Johnny Cash (Lofi)", "youtube_link": "https://www.youtube.com/watch?v=fRqDwQ-FFe4"}
            ]
        else:
            songs = get_songs(emotion, strategy)
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

@app.route('/recommend/fusion', methods=['POST'])
def recommend_fusion():
    try:
        data = request.get_json()
        text = data.get('text', '')
        image_data = data.get('image', '')
        strategy = data.get('strategy', 'match')
        client_metrics = data.get('clientMetrics', {})

        text_breakdown = None
        webcam_breakdown = None

        if text:
            _, _, text_breakdown = predict_text_emotion(text)
            if text_breakdown:
                text_breakdown = {k: float(v) for k, v in text_breakdown.items()}

        if image_data:
            try:
                image_data_clean = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data_clean)
                np_arr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                result = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='ssd'
                )

                emotions_dict = result[0]['emotion']
                if 'neutral' in emotions_dict:
                    emotions_dict['neutral'] = emotions_dict['neutral'] * 0.25

                # Apply client landmarks metrics in fusion mode
                emotions_dict = apply_client_metrics(emotions_dict, client_metrics)

                webcam_breakdown = {EMOTION_MAP.get(k, k): float(round(float(v), 1)) for k, v in emotions_dict.items()}
            except Exception as e:
                print("Fusion webcam error:", e)

        all_keys = set(list(le.classes_) + list(EMOTION_MAP.values()))

        fused_breakdown = {}
        for key in all_keys:
            val_text = float(text_breakdown.get(key, 0.0)) if text_breakdown else 0.0
            val_web = float(webcam_breakdown.get(key, 0.0)) if webcam_breakdown else 0.0

            if text_breakdown and webcam_breakdown:
                fused_breakdown[key] = float(round((val_text + val_web) / 2.0, 1))
            elif text_breakdown:
                fused_breakdown[key] = val_text
            elif webcam_breakdown:
                fused_breakdown[key] = val_web
            else:
                fused_breakdown[key] = 0.0

        if not text_breakdown and not webcam_breakdown:
            return jsonify({'error': 'No input provided for fusion'}), 400

        # Normalize fusion breakdown to sum up to 100% (cast to float to avoid numpy serialization errors)
        fused_sum = sum(fused_breakdown.values())
        if fused_sum > 0:
            fused_breakdown = {k: float(round((v / fused_sum) * 100.0, 1)) for k, v in fused_breakdown.items()}

        emotion = max(fused_breakdown, key=fused_breakdown.get)
        confidence = float(fused_breakdown[emotion])

        if confidence == 0:
            emotion = 'neutral'
            confidence = 100.0
            fused_breakdown['neutral'] = 100.0

        # LIME attributions for the text input (if provided) based on the final fused emotion
        lime_attributions = None
        if text:
            lime_attributions = explain_text_lime(text, emotion)

        # Override with exact songs suitable for custom sad reactions if strategy is 'match'
        is_sad_hand_on_cheek = client_metrics.get('isSadHandOnCheek', False)
        is_sad_lips_down = client_metrics.get('isSadLipsDown', False)
        if strategy == 'match' and is_sad_hand_on_cheek:
            songs = [
                {"song": "Po Nee Po (Tamil Lofi Flip)", "artist": "Anirudh Ravichander (Lofi)", "youtube_link": "https://www.youtube.com/watch?v=FPpaBN_Go_I"},
                {"song": "The Night We Met (English Lofi)", "artist": "Lord Huron (Lofi)", "youtube_link": "https://www.youtube.com/watch?v=WxA-agYHG4w"},
                {"song": "Kanave Kanave (Tamil Lofi Flip)", "artist": "Anirudh Ravichander (Lofi)", "youtube_link": "https://www.youtube.com/watch?v=7Bx6S1I9t1s"},
                {"song": "Fix You (English Lofi)", "artist": "Coldplay (Lofi)", "youtube_link": "https://www.youtube.com/watch?v=T42VNg6KhWo"}
            ]
        elif strategy == 'match' and is_sad_lips_down:
            songs = [
                {"song": "Kannaana Kanney (Tamil Lofi)", "artist": "D. Imman (Lofi)", "youtube_link": "https://www.youtube.com/watch?v=C0KquTdmuoM"},
                {"song": "Someone Like You (English Lofi)", "artist": "Adele (Lofi)", "youtube_link": "https://www.youtube.com/watch?v=grT_dzV0QUQ"},
                {"song": "Maruvaarthai (Tamil Lofi)", "artist": "Sid Sriram (Lofi)", "youtube_link": "https://www.youtube.com/watch?v=9uZpMnXKLq8"},
                {"song": "Hurt (English Lofi)", "artist": "Johnny Cash (Lofi)", "youtube_link": "https://www.youtube.com/watch?v=fRqDwQ-FFe4"}
            ]
        else:
            songs = get_songs(emotion, strategy)
        quote = random.choice(QUOTES.get(emotion, QUOTES['neutral']))
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'recommendations': songs,
            'method': 'fusion',
            'quote': quote,
            'breakdown': fused_breakdown,
            'lime_attributions': lime_attributions
        })

    except Exception as e:
        import traceback
        print("FUSION ERROR:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/recommend/direct/<emotion>', methods=['GET'])
def recommend_direct(emotion):
    strategy = request.args.get('strategy', 'match')
    songs = get_songs(emotion, strategy)
    quote = random.choice(QUOTES.get(emotion, QUOTES['neutral']))
    return jsonify({
        'emotion': emotion,
        'confidence': 100.0,
        'recommendations': songs,
        'quote': quote
    })

@app.route('/telemetry/export', methods=['POST'])
def telemetry_export():
    try:
        data = request.get_json() or {}
        history = data.get('history', [])
        
        import io
        import csv
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Time', 'Source Input', 'Detected Emotion', 'Confidence (%)', 'Mood Transition State'])
        
        for entry in history:
            writer.writerow([
                entry.get('time', ''),
                entry.get('source', ''),
                entry.get('emotion', ''),
                entry.get('confidence', ''),
                entry.get('swingLabel', '')
            ])
            
        csv_data = output.getvalue()
        return csv_data, 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': 'attachment; filename=harmony_mind_history.csv'
        }
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_pdf_report_buffer(history, stats):
    import io
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=45, leftMargin=45, topMargin=45, bottomMargin=45)
    
    styles = getSampleStyleSheet()
    
    primary_color = colors.HexColor('#6366f1')
    text_color = colors.HexColor('#1f2937')
    light_bg = colors.HexColor('#f3f4f6')
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=primary_color,
        spaceAfter=15,
        alignment=0
    )
    
    h1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=text_color,
        leading=14,
        spaceAfter=8
    )
    
    bold_style = ParagraphStyle(
        'BodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    story = []
    
    story.append(Paragraph("HarmonyMind – Emotion Study Analytics Report", title_style))
    story.append(Paragraph("This report provides an analytical review of your cognitive-emotional states tracked during your study session, along with customized audio recommendations suggested by our Deep Learning model.", body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("1. Focus & Stability Analytics", h1_style))
    
    stats_data = [
        [Paragraph("Metric", bold_style), Paragraph("Value", bold_style), Paragraph("Description", bold_style)],
        [Paragraph("Total Mood Scans", body_style), Paragraph(str(stats.get('count', len(history))), body_style), Paragraph("Number of emotion scans recorded during session", body_style)],
        [Paragraph("Dominant Emotion", body_style), Paragraph(str(stats.get('dominant', 'Neutral')).capitalize(), body_style), Paragraph("The emotion detected most frequently during session", body_style)],
        [Paragraph("Focus Stability Score", body_style), Paragraph(str(stats.get('stability', '100%')), body_style), Paragraph("Consistency of focus (higher means fewer abrupt mood swings)", body_style)],
    ]
    
    t = Table(stats_data, colWidths=[130, 85, 305])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), light_bg),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#fafafa')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("2. Telemetry History Log", h1_style))
    
    history_data = [
        [Paragraph("Timestamp", bold_style), Paragraph("Source", bold_style), Paragraph("Predicted Emotion", bold_style), Paragraph("Confidence", bold_style), Paragraph("Transition State", bold_style)]
    ]
    
    for entry in history:
        history_data.append([
            Paragraph(entry.get('time', ''), body_style),
            Paragraph(entry.get('source', ''), body_style),
            Paragraph(entry.get('emotion', '').capitalize(), body_style),
            Paragraph(f"{entry.get('confidence', '')}%", body_style),
            Paragraph(entry.get('swingLabel', ''), body_style),
        ])
        
    t_history = Table(history_data, colWidths=[80, 75, 110, 80, 175])
    t_history.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), light_bg),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#fafafa')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_history)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("3. Deep Learning Methodology Details", h1_style))
    story.append(Paragraph("<b>Model Architecture:</b> Module 1: Bidirectional LSTM network optimized for text semantic embeddings. Module 2: Deep Convolutional Neural Networks (DCNN via DeepFace/SSD) optimized for real-time facial expression analysis. The decision layer uses real-time multimodal fusion.", body_style))
    story.append(Paragraph("<b>Explainable AI (XAI):</b> Local attributions are generated in real-time using LIME (Local Interpretable Model-agnostic Explanations) showing token contributions, and OpenCV landmark detection to capture facial expressions.", body_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

@app.route('/telemetry/report', methods=['POST'])
def telemetry_report():
    try:
        from flask import send_file
        data = request.get_json() or {}
        history = data.get('history', [])
        stats = data.get('stats', {})
        
        buffer = generate_pdf_report_buffer(history, stats)
        
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='harmony_mind_report.pdf'
        )
    except Exception as e:
        import traceback
        print("PDF REPORT ERROR:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
