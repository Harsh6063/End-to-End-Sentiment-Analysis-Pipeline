from flask import Flask, request, jsonify
import joblib
import re

# Load the trained model and TF-IDF vectorizer
try:
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    sentiment_model = joblib.load('sentiment_model.pkl')
    print("Model and vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading model/vectorizer: {e}")
    exit(1)

# Initialize the Flask app
app = Flask(__name__)

# Preprocessing function (same as during training)
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# Define the predict endpoint
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    try:
        data = request.get_json()

        if not data or 'review_text' not in data:
            return jsonify({'error': 'Missing "review_text" field in request.'}), 400

    
        review_text = preprocess_text(data['review_text'])


        if not review_text:
            return jsonify({'error': 'Input text is empty after preprocessing.'}), 400

    
        review_vectorized = tfidf_vectorizer.transform([review_text])
        prediction = sentiment_model.predict(review_vectorized)[0]
        sentiment = 'positive' if prediction == 1 else 'negative'
    
        return jsonify({'sentiment_prediction': sentiment})

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
