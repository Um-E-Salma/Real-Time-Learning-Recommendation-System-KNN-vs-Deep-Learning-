import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

st.set_page_config(page_title="Learning Recommendation System", layout="centered")
st.title("📚 Real-Time Personalized Learning Recommendations")

model_type = st.sidebar.selectbox("Choose Recommendation Model", ["KNN", "Deep Learning"])
action_type = st.sidebar.selectbox("Action Type", ['enter', 'respond', 'submit'])
user_answer = st.sidebar.selectbox("User Answer", ['Correct', 'Incorrect', 'No Answer'])
platform = st.sidebar.selectbox("Platform", ['mobile', 'web', 'tablet'])
hour = st.sidebar.slider("Hour", 0, 23, 12)
day = st.sidebar.selectbox("Day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

mappings = {
    'action_type': {'enter': 0, 'respond': 1, 'submit': 2},
    'user_answer': {'Correct': 1, 'Incorrect': 0, 'No Answer': -1},
    'platform': {'mobile': 0, 'web': 1, 'tablet': 2},
    'day': {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6},
}

input_data = np.array([[
    hour,
    mappings['day'][day],
    mappings['action_type'][action_type],
    mappings['platform'][platform],
    0  # source_encoded (not used)
]])

if st.sidebar.button("Submit"):
    if model_type == "KNN":
        model = joblib.load("models/knn_model.pkl")
        prediction = model.predict(input_data)[0]
    else:
        model = tf.keras.models.load_model("models/deeplearning_model.keras")
        prediction = model.predict(input_data).round()[0][0]

    if prediction == 0:
        st.info("📘 Recommendation: Practice exercises or short videos.")
    else:
        st.success("📗 Recommendation: Deep articles or advanced content.")

    st.markdown("---")
    st.subheader("📊 Model Comparison Table")
    df = pd.read_csv("model_comparison.csv")
    st.dataframe(df)

    st.markdown("---")
    st.subheader("📈 Activity Graphs")
    st.image("hourly_activity.png", caption="Activity by Hour")
    st.image("weekly_activity.png", caption="Activity by Day of the Week")

    st.markdown("---")
    st.subheader("📚 Literature Review")
    st.markdown("""
    - KNN is a traditional instance-based learning algorithm used in various recommendation systems (Chapter

    Primer on artificial intelligence
    2022, Mobile Edge Artificial Intelligence
    Yuanming Shi, ... Yong Zhou).
    - Deep learning offers flexibility and scalability, with neural networks adapting better to non-linear patterns in user data.
    """)
    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ using TensorFlow and Streamlit")


