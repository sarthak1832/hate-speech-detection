import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Load dataset
df = pd.read_csv('data/labeled_data.csv')

# Show class distribution
print("Class distribution:")
print(df['class'].value_counts())

# Features and target
X = df['tweet']
y = df['class']

# Vectorizer with n-grams and min/max df to reduce noise
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    min_df=5,
    max_df=0.7,
    ngram_range=(1, 2)
)
X_vec = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# Train model with balanced class weights to handle imbalance
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Print classification report for detailed metrics
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Hate Speech', 'Offensive Language', 'Neither']))

# Save model and vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Model trained and saved!")