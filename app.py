import streamlit as st
import joblib
import re

# ---------- Helpers ----------

def clean_text(text: str) -> str:
    """
    Basic text cleaning: lowercase, remove punctuation, extra spaces.
    This should match what we did in the notebook.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_artifacts():
    """
    Load the trained model, TF-IDF vectorizer, and label encoder.
    Cached so they are loaded only once.
    """
    model = joblib.load("models/lr_model.pkl")
    tfidf = joblib.load("models/tfidf.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return model, tfidf, label_encoder


# ---------- Load model + vectorizer + encoder ----------

model, tfidf, label_encoder = load_artifacts()


# ---------- Streamlit UI ----------

st.title("🔐 Security Ticket Intent Classifier")

st.write(
    """
    Type a short description of an IT security ticket, and the model will predict its category.
    
    **Example:**  
    - "I can't log into VPN from home"  
    - "Not receiving MFA codes"  
    - "Email says my password was found in a breach"
    """
)

user_text = st.text_area("Enter a ticket description:", height=120)

if st.button("Predict"):
    if not user_text.strip():
        st.warning("Please enter a description first.")
    else:
        # 1) Clean text
        cleaned = clean_text(user_text)

        # 2) Vectorize
        X_input = tfidf.transform([cleaned])

        # 3) Predict
        pred_encoded = model.predict(X_input)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        # 4) Optional: prediction probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_input)[0]
            classes = label_encoder.classes_
            prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
        else:
            prob_dict = None

        st.success(f"Predicted category: **{pred_label}**")

        if prob_dict is not None:
            st.write("Prediction confidence:")

        # Create a DataFrame for plotting
        prob_df = pd.DataFrame({
            "Category": list(prob_dict.keys()),
            "Probability": list(prob_dict.values())
        }).set_index("Category")

        st.bar_chart(prob_df)

