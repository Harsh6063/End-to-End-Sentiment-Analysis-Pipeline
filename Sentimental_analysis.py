from datasets import load_dataset
import pandas as pd
import re
from bs4 import BeautifulSoup
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report



# Load the IMDb dataset
dataset = load_dataset("imdb")


#This Dataset Contains two columns TEXT in which the review 
# is written and the label column which indicates if its a 
# Positive Review or not (0:-ve,1:+ve)

train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

def clean_text(text):
    text = text.lower()
    
    text = BeautifulSoup(text, "html.parser").get_text()

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Applying the cleaning function 
train_df['cleaned_review'] = train_df['text'].apply(clean_text)
test_df['cleaned_review'] = test_df['text'].apply(clean_text)

train_df.drop_duplicates(inplace=True)
test_df.drop_duplicates(inplace=True)
print("Train Data Head:")
print(train_df[['text', 'cleaned_review']].head())

print("\nTest Data Head:")
print(test_df[['text', 'cleaned_review']].head())

#Some EDA

sentiment_counts = train_df['label'].value_counts()

# 2. Average review length for positive vs negative
train_df['review_length'] = train_df['cleaned_review'].apply(len)
avg_review_length = train_df.groupby('label')['review_length'].mean()

# 3. Create a plot for the distribution of sentiments
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=train_df, palette='viridis')
plt.title('Number of Reviews per Sentiment')
plt.ylabel('Number of Reviews')
plt.xlabel('Sentiment')
plt.show()

# 4.  Plot average review length for positive vs negative
avg_review_length.plot(kind='bar', figsize=(8, 6), color=['green', 'red'])
plt.title('Average Review Length for Positive vs Negative Reviews')
plt.ylabel('Average Review Length')
plt.xlabel('Sentiment')
plt.xticks(rotation=0)
plt.show()



# Show stats
print("Sentiment Distribution:")
print(sentiment_counts)

print("\nAverage Review Length for Positive and Negative Sentiments:")
print(avg_review_length)

#Splitting X_Train and y_Train

X_train = train_df['cleaned_review']
y_train = train_df['label']
X_test = test_df['cleaned_review']
y_test = test_df['label']

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize and train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test_tfidf)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))


# Saving the Models for app.py to use

import joblib

# Save the TF-IDF vectorizer and Naive Bayes model
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(nb_classifier, 'sentiment_model.pkl')

print("Model and vectorizer saved!")
    