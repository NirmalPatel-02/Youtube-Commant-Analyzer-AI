from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from youtube_comment_downloader import YoutubeCommentDownloader

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Enable zip and len in Jinja templates
templates.env.globals['zip'] = zip
templates.env.globals['len'] = len

# Load model and artifacts
model = tf.keras.models.load_model('model/toxic_classifier_gru.keras')
with open('model/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
with open('model/thresholds.pickle', 'rb') as f:
    thresholds = pickle.load(f)  # list of 6 thresholds

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
MAX_LEN = 150

# Text cleaning (same as training)
def clean_text(text):
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    text = re.sub(r'\={2,}\s*(.+?)\s*\={2,}', ' ', text)
    text = re.sub(r'\={2,}', ' ', text)
    text = re.sub(r'\[\[.*?\]\]', ' ', text)
    text = re.sub(r'\{\{.*?\}\}', ' ', text)
    text = re.sub(r'http[s]?\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    contractions = {
        "ain't": "am not", "aren't": "are not", "can't": "cannot", "could've": "could have",
        "don't": "do not", "doesn't": "does not", "didn't": "did not", "he's": "he is",
        "i'm": "i am", "it's": "it is", "isn't": "is not", "let's": "let us",
        "she's": "she is", "that's": "that is", "there's": "there is",
        "they're": "they are", "wasn't": "was not", "we're": "we are",
        "weren't": "were not", "won't": "will not", "you're": "you are", "you've": "you have"
    }
    for contr, full in contractions.items():
        text = text.replace(contr, full)
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    return text.strip()

# Batch prediction
def predict_batch(texts):
    cleaned = [clean_text(t) for t in texts]
    seq = tokenizer.texts_to_sequences(cleaned)
    seq = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    probs = model.predict(seq, verbose=0)
    
    results = []
    for prob in probs:
        flags = [labels[i] for i in range(6) if prob[i] > thresholds[i]]
        results.append(flags if flags else ["clean"])
    return results

# YouTube comment downloader (2025 working)
downloader = YoutubeCommentDownloader()

def get_youtube_comments(video_url, max_comments=300):
    try:
        video_id = None
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
        ]
        for pattern in patterns:
            match = re.search(pattern, video_url)
            if match:
                video_id = match.group(1)
                break
        if not video_id:
            return ["Invalid YouTube URL"]

        comments = []
        for comment in downloader.get_comments(video_id):
            text = comment.get('text', '').strip()
            if text:
                comments.append(text)
            if len(comments) >= max_comments:
                break
        return comments if comments else ["No comments found (disabled or private)"]
    except Exception as e:
        return [f"Error fetching comments: {str(e)}"]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def analyze(request: Request, youtube_url: str = Form(...)):
    if not youtube_url.strip():
        return templates.TemplateResponse("index.html", {"request": request, "error": "Enter a YouTube URL"})

    comments = get_youtube_comments(youtube_url, max_comments=300)

    if not comments or isinstance(comments[0], str) and ("Error" in comments[0] or "Invalid" in comments[0]):
        error_msg = comments[0] if comments else "No comments found"
        return templates.TemplateResponse("index.html", {"request": request, "error": error_msg})

    predictions = predict_batch(comments)

    # Count labels
    counts = {label: 0 for label in labels}
    counts["clean"] = 0
    for pred in predictions:
        if "clean" in pred:
            counts["clean"] += 1
        for label in pred:
            if label in counts:
                counts[label] += 1

    return templates.TemplateResponse("index.html", {
        "request": request,
        "url": youtube_url,
        "total_comments": len(comments),
        "counts": counts,
        "comments_sample": comments[:10],
        "predictions_sample": predictions[:10]
    })