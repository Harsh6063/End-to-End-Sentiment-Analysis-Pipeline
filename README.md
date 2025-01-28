# End-to-End-Sentiment-Analysis-Pipeline

# Sentiment Analysis Flask App

This project is a sentiment analysis system built using Flask, Naive Bayes, and TF-IDF vectorization. It includes a machine learning pipeline for training a sentiment classification model and serving predictions via a RESTful API.

---

## Project Setup

### Prerequisites
- Python 3.8+
- Pip (Python package manager)

### Installation Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/End-to-End-Sentiment-Analysis-Pipeline.git
    cd End-to-End-Sentiment-Analysis-Pipeline.git

### 2 . Install dependencies```bash
    pip install -r requirements.txt


 ### 3. Data Acquisition
The dataset used in this project is the IMDB Reviews Dataset. It was downloaded using the Hugging Face Datasets library:
    ```bash

    from datasets import load_dataset
    dataset = load_dataset("imdb")
The dataset was split into training and testing datasets. The reviews were cleaned by:

- i.)Lowercasing text
- ii.)Removing HTML tags
- iii.)Optionally removing punctuation

### 4. Run the training script to train the Naive Bayes model on the IMDB dataset:

python sentimental_analysis.py

This script:
Loads and preprocesses the dataset
Vectorizes the text using TF-IDF
Trains a Naive Bayes classifier
Saves the trained model to two files (tfidf_vectorizer.pkl) and  (sentiment_model.pkl)

now for the flask app
Run the Flask app:
python app.py
This starts the server at http://127.0.0.1:5000/.

To test the /predict endpoint, use the following curl command:
     ```bash
    
    curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"review_text\": \"This is an amazing movie!\"}"

### 5. Model Summary

The model is based on a **Naive Bayes classifier** trained using **TF-IDF vectorization**.

#### Key Metrics
After training the model on the IMDB dataset, the following metrics were achieved on the test set:

- **Accuracy**: 0.8417
- **Precision**: 0.8519
- **Recall**: 0.8283
- **F1 Score**: 0.8399

#### Approach Summary
```plaintext
1. Preprocessing:
   - The data was cleaned to remove unwanted characters, HTML tags, and punctuation.

2. Vectorization:
   - Text data was transformed into numerical features using TF-IDF.

3. Model:
   - A Multinomial Naive Bayes classifier was chosen for its simplicity and speed.

4. Deployment:
   - The model was deployed using Flask, providing a RESTful API for predictions.






