import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page setup
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊")
st.title("😊😠 Product Review Sentiment Analyzer")
st.write("Classify reviews as Positive, Neutral, or Negative using ML")

# Sample dataset (small for demo)
@st.cache_data
def load_data():
    data = pd.DataFrame({
        "review": [
            "I love this product! Amazing quality.",
            "It's okay, but could be better.",
            "Terrible experience. Would not buy again.",
            "Great value for money.",
            "Not what I expected.",
            "Perfect! Exactly as described.",
            "Waste of money.",
            "Average product, nothing special.",
            "Highly recommend!",
            "Disappointed with the purchase."
        ],
        "sentiment": ["positive", "neutral", "negative", "positive", "neutral",
                     "positive", "negative", "neutral", "positive", "negative"]
    })
    return data

data = load_data()

# Show dataset
if st.checkbox("Show sample reviews"):
    st.write(data)

# Preprocess data
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(data["review"])
y = data["sentiment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Show accuracy
st.metric("Model Accuracy", f"{accuracy:.0%}")

# User input for prediction
st.subheader("🔍 Try It Yourself")
user_review = st.text_area("Enter a product review:", "This product is great!")

if st.button("Predict Sentiment"):
    # Vectorize input
    review_vec = vectorizer.transform([user_review])
    prediction = model.predict(review_vec)[0]
    proba = model.predict_proba(review_vec).max()

    # Display result with emoji
    if prediction == "positive":
        st.success(f"😊 Positive ({proba:.0%} confidence)")
    elif prediction == "neutral":
        st.info(f"😐 Neutral ({proba:.0%} confidence)")
    else:
        st.error(f"😠 Negative ({proba:.0%} confidence)")

# Footer
st.markdown("---")
st.caption("Internship Project | Sentiment Analysis with Naive Bayes")
