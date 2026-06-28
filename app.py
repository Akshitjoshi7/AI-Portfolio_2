import streamlit as st
from dotenv import load_dotenv
import os
import requests

# Load environment variables from .env file
load_dotenv()

# Get GROQ API key from environment variable
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Hardcoded transcript for now
transcript = "AI is changing the world. Large language models are becoming the backbone of modern software. Every startup is racing to build AI products."

# Function to send transcript to Groq API and get tweets
def generate_tweets(transcript):
    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        'prompt': transcript,
        'model': 'llama-3.3-70b-versatile',
        'max_tokens': 280
    }
    response = requests.post('https://api.groq.com/v1/generate', headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['text'].split('\n')
    else:
        st.error("Failed to generate tweets")
        return []

# Streamlit app
st.title("YouTube to Twitter Thread Generator")

# Text input box for YouTube URL
youtube_url = st.text_input("Enter YouTube URL:")

# Button: "Generate Thread"
if st.button("Generate Thread"):
    if youtube_url:
        # For now, use hardcoded transcript
        tweets = generate_tweets(transcript)
        
        # Display tweets numbered 1-10 on screen
        for i, tweet in enumerate(tweets[:10], start=1):
            st.write(f"{i}. {tweet.strip()}")
    else:
        st.warning("Please enter a YouTube URL")
