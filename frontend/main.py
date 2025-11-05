import streamlit as st

def get_prediction(input_text):
    return {"prediction": "No prediction"}

# Streamlit page configuration
st.set_page_config(page_title="Tweet Classifier", layout="wide")

# Streamlit UI components
st.title("Classify your tweet")

# User inputs the tweet
tweet_input = st.text_input("Enter your tweet", "")

# Button to trigger prediction
if st.button("Classify Tweet"):
    # Get prediction
    prediction = get_prediction(tweet_input)

    # Display the prediction
    st.write("Prediction:", prediction)


