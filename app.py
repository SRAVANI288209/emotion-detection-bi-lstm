
import streamlit as st
import tensorflow as tf
import pickle
import re
import html
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==============================
# Load Model
# ==============================
model = tf.keras.models.load_model(
    "/kaggle/working/emotion_bilstm_best.keras"
)

# ==============================
# Load Tokenizer
# ==============================
with open("/kaggle/working/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ==============================
# Load Label Encoder
# ==============================
with open("/kaggle/working/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

max_length = 31


# ==============================
# Negation Handling
# ==============================
def handle_negation(text):
    words = text.split()
    result = []
    negation = False

    negation_words = {
        "not", "no", "never", "dont", "cannot", "cant"
    }

    for word in words:
        result.append(word)

        if word in negation_words:
            negation = True

        elif negation:
            result[-1] = word + "_NEG"
            negation = False

    return " ".join(result)


# ==============================
# Text Preprocessing
# ==============================
def preprocess_text(text):

    text = text.lower()

    text = html.unescape(text)

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z0-9\s❤️]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    text = re.sub(
        r"\d+",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Same negation handling used during training
    text = handle_negation(text)

    return text


# ==============================
# Streamlit UI
# ==============================
st.title("😊 Emotion Detection using BiLSTM")

st.write(
    "Enter a sentence and the model will predict the emotion."
)

user_text = st.text_area(
    "Enter your text:"
)


# ==============================
# Prediction
# ==============================
if st.button("Predict Emotion"):

    if user_text.strip() == "":
        st.warning("Please enter some text.")

    else:

        cleaned_text = preprocess_text(user_text)

        sequence = tokenizer.texts_to_sequences(
            [cleaned_text]
        )

        padded_sequence = pad_sequences(
            sequence,
            maxlen=max_length,
            padding="post",
            truncating="post"
        )

        prediction = model.predict(
            padded_sequence,
            verbose=0
        )

        predicted_class = prediction.argmax(axis=1)[0]

        predicted_emotion = label_encoder.inverse_transform(
            [predicted_class]
        )[0]

        confidence = (
            prediction[0][predicted_class] * 100
        )

        st.success(
            f"Predicted Emotion: {predicted_emotion}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

print("===== app.py UPDATED SUCCESSFULLY =====")
