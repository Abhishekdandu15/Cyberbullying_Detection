import streamlit as st
import re
import string
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# User Authentication
users = {"admin": "password123"}  # Simple user authentication

def login_page():
    st.title("User Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            if username in users and users[username] == password:
                st.session_state["authenticated"] = True
                st.success("Login successful!")
            else:
                st.error("Invalid username or password")
    with col2:
        if st.button("Register"):
            st.session_state["register"] = True

def register_page():
    st.title("User Registration")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Register"):
            if new_username and new_password:
                users[new_username] = new_password
                st.success("Registration successful! Please log in now.")
                del st.session_state["register"]
                st.session_state["authenticated"] = False
    with col2:
        if st.button("Go to Login"):
            del st.session_state["register"]
            st.rerun()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def train_model(df):
    df["cleaned_tweet"] = df["tweet_text"].apply(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(df["cleaned_tweet"], df["label"], test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    joblib.dump(model, "model_nb.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    
    return accuracy, report, model, vectorizer, X_test, y_test

def main():
    if "register" in st.session_state:
        register_page()
        return
    
    if "authenticated" not in st.session_state:
        login_page()
        return
    
    st.title("Cyberbullying Detection System")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding="latin1")
        df["label"] = df["cyberbullying_type"].apply(lambda x: 0 if x == "not_cyberbullying" else 1)
        
        st.subheader("Dataset Preview")
        st.write(df.head(4))  # Show first 4 rows before training
        
        if st.button("Train Model"):
            accuracy, report, model, vectorizer, X_test, y_test = train_model(df)
            st.write(f"Model Accuracy: {accuracy}")
            
            # Show Graphs
            fig, ax = plt.subplots()
            sns.countplot(x=df["label"], ax=ax)
            st.pyplot(fig)
            
            fig, ax = plt.subplots()
            df["label"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            st.pyplot(fig)
            
            st.write("Classification Report:")
            st.text(report)
            
            # Visualizing predictions
            predictions = model.predict(vectorizer.transform(X_test))
            results_df = pd.DataFrame({"Tweet": X_test, "Actual": y_test, "Predicted": predictions})
            st.write("Prediction Results")
            st.write(results_df.head(10))
    
    user_input = st.text_area("Enter tweet to analyze:")
    if st.button("Detect"):
        try:
            model = joblib.load("model_nb.pkl")
            vectorizer = joblib.load("vectorizer.pkl")
            cleaned_input = clean_text(user_input)
            input_tfidf = vectorizer.transform([cleaned_input])
            prediction = model.predict(input_tfidf)[0]
            result = "🚨 Cyberbullying Detected!" if prediction == 1 else "✅ Not Cyberbullying"
            st.subheader(result)
        except FileNotFoundError:
            st.error("Please train the model first.")

if __name__ == "__main__":
    main()
