# main.py
import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from s1_logic import ColorTranslator, DataSerializer

app = FastAPI()

# --- Endpoint 1: Local Logic (Color Conversion) ---
@app.get("/rgb-to-yuv")
def convert_color(r: int, g: int, b: int):
    # This runs inside this API container using your s1_logic module
    y, u, v = ColorTranslator.rgb_to_yuv(r, g, b)
    return {"rgb": [r, g, b], "yuv": [y, u, v]}

# --- Endpoint 2: Local Logic (RLE Encoding) ---
class ListData(BaseModel):
    data: list

@app.post("/encode-rle")
def encode_list(payload: ListData):
    encoded = DataSerializer.run_length_encoding(payload.data)
    return {"original": payload.data, "encoded": encoded}

# --- Endpoint 3: Interaction with FFMPEG Docker ---
# This satisfies the requirement: "launch your API and it will call the FFMPEG docker" 
@app.post("/convert-video")
def convert_video(video_name: str):
    ffmpeg_url = "http://ffmpeg-service:5000/convert"  # Calling the other container
    try:
        response = requests.post(ffmpeg_url, json={"filename": video_name})
        return response.json()
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="FFMPEG Service unavailable")