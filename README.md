# YouTube to Twitter Thread Generator

🔗 **Live Demo:** https://huggingface.co/spaces/Akshitjoshi9897/yt-thread-generator


A Streamlit app that turns a YouTube video into a concise 10-tweet Twitter/X thread. The app downloads audio from a YouTube URL, transcribes it with Whisper, and uses Groq's LLM API to generate an engaging tweet thread from the transcript.

![Demo](demo.gif)

## Features

- YouTube audio transcription from pasted video URLs
- AI-generated 10-tweet Twitter/X threads
- Groq LLM integration for fast text generation
- Fallback transcript if YouTube download or transcription fails
- Simple Streamlit interface

## Tech Stack

- Python
- Streamlit
- Whisper
- yt-dlp
- Groq
- LangChain

## Setup

1. Clone the repository:

```bash
git clone https://github.com/Akshitjoshi7/AI-Portfolio.git
cd AI-Portfolio/project-1-yt-thread
```

2. Install dependencies:

```bash
pip install streamlit python-dotenv groq openai-whisper yt-dlp langchain
```

3. Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

4. Run the app:

```bash
streamlit run app.py
```

## Screenshot

Add a screenshot of the running app here.

```markdown
![Screenshot](screenshot.png)
```
