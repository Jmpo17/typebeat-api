from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import tempfile

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok", "message": "TypeBeat API running"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=None, mono=True)

        # Extract tempo and key
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_idx = chroma.mean(axis=1).argmax()
        keys = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        key = keys[key_idx]

        # Rough “energy” = RMS
        rms = librosa.feature.rms(y=y)
        energy = float(np.mean(rms))

        # Placeholder tags (later replace with ML + Musiio + Spotify)
        tags = ["drill", "dark", "street"]
        artists = ["Central Cee", "Headie One"]

        return JSONResponse({
            "bpm": float(tempo),
            "key": key,
            "energy": energy,
            "tags": tags,
            "artists": artists
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
