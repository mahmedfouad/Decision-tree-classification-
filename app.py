import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="PlayTennis Predictor", page_icon="🎾")

@st.cache_resource
def load_model():
    return joblib.load("playtennis_dt_gini.joblib")

package = load_model()
model = package["model"]
encoders = package["encoders"]
feature_columns = package["feature_columns"]

st.title("🎾 Play Tennis Predictor")
st.write("Select today's conditions to predict whether tennis will be played.")

# Build one dropdown per feature, using the categories the model was actually trained on
user_input = {}
for col in feature_columns:
    options = list(encoders[col].classes_)          # e.g. [False, True] or ['High', 'Normal']
    display_options = [str(o) for o in options]      # what the user sees
    choice_display = st.selectbox(col, display_options)
    # map the displayed string back to the original value type the encoder expects
    choice_actual = options[display_options.index(choice_display)]
    user_input[col] = choice_actual

if st.button("Predict"):
    encoded_row = [[encoders[col].transform([user_input[col]])[0] for col in feature_columns]]
    input_df = pd.DataFrame(encoded_row, columns=feature_columns)
    pred = model.predict(input_df)
    label = encoders["PlayTennis"].inverse_transform(pred)[0]

    if label == "Yes":
        st.success(f"Prediction: **{label}** 🎾 — good day for tennis!")
    else:
        st.error(f"Prediction: **{label}** — maybe stay indoors.")