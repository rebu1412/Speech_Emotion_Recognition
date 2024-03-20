import streamlit as st
from utils import predict_audio

def main():
    st.title("Speech Emotion Recognition")

    uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/mp3', start_time=0)

        if st.button("Predict"):
            emotion = predict_audio(uploaded_file)
            st.write("Emotion:", emotion)
            
if __name__ == "__main__":
    main()