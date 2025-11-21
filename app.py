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
templates.env.globals['zip'] = zip
templates.env.globals['len'] = len

model = tf.keras.models.load_model('model/toxic_classifier_gru.keras')
with open('model/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
with open('model/thresholds.pickle', 'rb') as f:
    thresholds = pickle.load(f)

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
MAX_LEN = 150
downloader = YoutubeCommentDownloader()

def clean_text(text):
    if not isinstance(text, str): text = ""
    text = text.lower()
    text = re.sub(r'\={2,}\s*(.+?)\s*\={2,}', ' ', text)
    text = re.sub(r'\={2,}', ' ', text)
    text = re.sub(r'\[\[.*?\]\]', ' ', text)
    text = re.sub(r'\{\{.*?\}\}', ' ', text)
    text = re.sub(r'http[s]?\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    contractions = {"ain't": "am not", "aren't": "are not", "can't": "cannot", "don't": "do not", "it's": "it is", "i'm": "i am", "80": "you are", "he's": "he is", "she's": "she is", "that's": "that is", "there's": "there is", "they're": "they are", "we're": "we are", "won't": "will not"}
    for c, f in contractions.items():
        text = text.replace(c, f)
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    return text.strip()

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

def get_youtube_comments(url, max_comments=500):
    try:
        video_id = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
        if not video_id: return None
        video_id = video_id.group(1)
        comments = []
        for c in downloader.get_comments(video_id):
            text = c.get('text', '').strip()
            if text: comments.append(text)
            if len(comments) >= max_comments: break
        return comments or None
    except Exception as e:
        print("Error:", e)
        return None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, youtube_url: str = Form(...)):
    if not youtube_url.strip():
        return templates.TemplateResponse("index.html", {"request": request, "error": "Please enter a valid URL"})

    comments = get_youtube_comments(youtube_url, 500)
    if not comments:
        return templates.TemplateResponse("index.html", {"request": request, "error": "No comments found or invalid video"})

    predictions = predict_batch(comments)
    total = len(comments)

    counts = {label: 0 for label in labels}
    counts["clean"] = 0
    comment_by_label = {label: [] for label in labels + ["clean"]}

    for comment, pred in zip(comments, predictions):
        if "clean" in pred:
            counts["clean"] += 1
            comment_by_label["clean"].append(comment)
        for label in pred:
            if label != "clean":
                counts[label] += 1
                comment_by_label[label].append(comment)

    for key in comment_by_label:
        comment_by_label[key] = comment_by_label[key][:10]

    toxic_pct = round((total - counts["clean"]) / total * 100, 2)
    if toxic_pct <= 5: heat = "Looking Good"
    elif toxic_pct <= 15: heat = "Ok i guess.. Not that Bad"
    elif toxic_pct <= 30: heat = "Moderate"
    elif toxic_pct <= 50: heat = "This is looking Bad"
    else: heat = "Extremely Bad"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "url": youtube_url,
        "total": total,
        "counts": counts,
        "top_comments": comment_by_label,
        "heat_index": heat,
        "heat_color": "green" if toxic_pct <= 15 else "yellow" if toxic_pct <= 40 else "red",
        "chart_data": {
            "bar": [counts["clean"]] + [counts[l] for l in labels],
            "pie": [counts["clean"], total - counts["clean"]]
        }
    })