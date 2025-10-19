from flask import Flask, render_template, request, jsonify
import re
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Initialize Flask app
app = Flask(__name__)

# Download WordNet
nltk.download('wordnet')

# Load and clean dataset
df = pd.read_csv("dataset.csv")
df = df[df['label'].isin(['0', '1'])].copy()
df.drop(columns=['Unnamed: 2', 'Unnamed: 3'], errors='ignore', inplace=True)
df = df[df['headline'].notna()]

# Clean and preprocess headlines
df['label'] = df['label'].astype(int)
df['headline'] = df['headline'].astype(str).str.strip()
df.drop_duplicates(subset=['headline'], inplace=True)
df['headline'] = df['headline'].str.replace(r'@\w+', '', regex=True)
df['headline'] = df['headline'].str.replace(r'http\S+|www\.\S+', '', regex=True)
df['headline'] = df['headline'].str.replace("[^a-zA-Z#]", " ", regex=True)

# Prepare data
X = df['headline']
y = df['label']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# Train XGBoost model
model = XGBClassifier(eval_metric='logloss', n_estimators=50, max_depth=5)
model.fit(X_train, y_train)

# Expand bullying keywords using WordNet
base_keywords = ["idiot", "stupid", "dumb", "ugly", "hate", "loser", "kill", "fat", "die", "freak"]
def expand_with_synonyms(words, max_synonyms=5):
    expanded = set(words)
    stop_synonyms = {"go", "move", "walk", "exit", "leave", "pass"} # Avoid common verbs
    for word in words:
        count = 0
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != word and synonym.isalpha():
                    expanded.add(synonym)
                    count += 1
                    if count >= max_synonyms:
                        break
            if count >= max_synonyms:
                break
    return list(expanded)

bullying_keywords = list(set(expand_with_synonyms(base_keywords)))
bullying_keywords = [w.strip().lower() for w in bullying_keywords if w.isalpha()]

# Helper to highlight keywords in the input
def highlight_bullying_words(text, keywords):
    def replacer(match):
        return f'<mark class="bg-danger text-white">{match.group(0)}</mark>'
    
    # Sort by length to avoid partial matching short words
    sorted_keywords = sorted(keywords, key=len, reverse=True)
    for word in sorted_keywords:
        pattern = r'\b' + re.escape(word) + r'\b'
        text = re.sub(pattern, replacer, text, flags=re.IGNORECASE)
    return text

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = data.get('text', '').strip()

    if not user_input:
        return jsonify({"error": "Empty input"})

    transformed_input = vectorizer.transform([user_input])
    prediction = int(model.predict(transformed_input)[0])
    probability = float(model.predict_proba(transformed_input)[0][1])
    bullying_percentage = round(probability * 100, 2) if prediction == 1 else 0.0

    # Strong keyword override (ensure max confidence for known bullying terms)
    strong_words = {"kill", "die", "loser", "idiot", "stupid", "dumb", "fuck", "hit", "ugly", "freak"}
    lower_text = user_input.lower()
    if any(word in lower_text for word in strong_words):
        prediction = 1
        bullying_percentage = max(bullying_percentage, 100.0)

    # Simple sentiment detection
    sentiment = "Neutral"
    if any(word in lower_text for word in ["happy", "good", "love", "great", "nice"]):
        sentiment = "Positive"
    elif any(word in lower_text for word in ["bad", "sad", "angry", "hate", "terrible"]):
        sentiment = "Negative"

    # Highlight bullying words
    highlighted = highlight_bullying_words(user_input, bullying_keywords)

    return jsonify({
        'prediction': prediction,
        'bullying_percentage': bullying_percentage,
        'highlighted': highlighted,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
