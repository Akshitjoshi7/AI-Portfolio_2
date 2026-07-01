import streamlit as st
from dotenv import load_dotenv
import os
import glob
import tempfile

# Load environment variables from .env file
load_dotenv()

# Get GROQ API key from environment variable
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Hardcoded transcript for now
transcript = "AI is changing the world. Large language models are becoming the backbone of modern software. Every startup is racing to build AI products."

def is_ffmpeg_missing_error(error):
    error_text = str(error).lower()
    return "winerror 2" in error_text or "ffmpeg" in error_text

def transcribe_youtube(youtube_url):
    try:
        import whisper
        from yt_dlp import YoutubeDL

        with tempfile.TemporaryDirectory() as temp_dir:
            ydl_opts = {
                'format': 'worstaudio/worst',
                'outtmpl': 'audio.%(ext)s',
                'paths': {'home': temp_dir},
                'quiet': True,
                'no_warnings': True,
            }

            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                audio_path = ydl.prepare_filename(info)

            if not os.path.exists(audio_path):
                audio_files = glob.glob(os.path.join(temp_dir, 'audio.*'))
                if not audio_files:
                    raise FileNotFoundError("Downloaded audio file was not found")
                audio_path = audio_files[0]

            model = whisper.load_model("tiny")
            try:
                result = model.transcribe(audio_path)
            except Exception as e:
                if is_ffmpeg_missing_error(e):
                    st.error("Please install ffmpeg from https://ffmpeg.org/download.html")
                raise

            text = result.get("text", "").strip()

            if not text:
                raise ValueError("Whisper returned an empty transcript")

            return text
    except Exception as e:
        st.warning(f"Using fallback transcript because transcription failed: {e}")
        return transcript

# Function to send transcript to Groq API and get tweets
def generate_tweets(transcript):
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f"""Convert this transcript into a 10-tweet Twitter thread.
Number each tweet 1-10.
Each tweet must be under 280 characters.
Make it engaging and insightful.

Transcript: {transcript}"""
            }
        ]
    )
    result = response.choices[0].message.content
    tweets = [line.strip() for line in result.split('\n') if line.strip()]
    return tweets

# Streamlit app
st.title("YouTube to Twitter Thread Generator")

# Text input box for YouTube URL
youtube_url = st.text_input("Enter YouTube URL:")

# Button: "Generate Thread"
if st.button("Generate Thread"):
    if youtube_url:
        with st.spinner("Downloading and transcribing..."):
            real_transcript = transcribe_youtube(youtube_url)

        tweets = generate_tweets(real_transcript)
        
        # Display tweets numbered 1-10 on screen
        for i, tweet in enumerate(tweets[:10], start=1):
            st.write(f"{i}. {tweet.strip()}")
    else:
        st.warning("Please enter a YouTube URL")
