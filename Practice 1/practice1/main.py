import os
import shutil
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List

# Import your logic
from s1_logic import ColorTranslator, DataSerializer, FFmpegAuto, DCT_Converter, DWT_Converter

app = FastAPI(title="Multimedia API", description="API for Seminar 1 & Dockerization")

# Helper to save uploaded files temporarily
def save_upload(upload_file: UploadFile) -> str:
    path = f"temp_{upload_file.filename}"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return path

# --- 1. BASIC ENDPOINTS ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Multimedia Processing API"}

@app.get("/rgb-to-yuv")
def convert_color(r: int, g: int, b: int):
    y, u, v = ColorTranslator.rgb_to_yuv(r, g, b)
    return {"rgb": [r, g, b], "yuv": [y, u, v]}

class ListData(BaseModel):
    data: list

@app.post("/encode-rle")
def encode_list(payload: ListData):
    encoded = DataSerializer.run_length_encoding(payload.data)
    return {"original": payload.data, "encoded": encoded}

# --- 2. DOCKER INTERACTION (The Assignment Requirement) ---
@app.post("/convert-video")
def convert_video(video_name: str):
    """
    Triggers the separate FFMPEG Docker Container.
    """
    ffmpeg_url = "http://ffmpeg-service:5000/convert"
    try:
        response = requests.post(ffmpeg_url, json={"filename": video_name})
        return response.json()
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="FFMPEG Service unavailable")

# --- 3. SEMINAR 1 IMAGE LOGIC ---

@app.post("/serpentine-read")
def serpentine_scan(file: UploadFile = File(...)):
    """
    Exercise 4: Uploads an image and returns pixels in serpentine order.
    """
    input_path = save_upload(file)
    try:
        pixels = DataSerializer.serpentine_read(input_path)
        # Limit response size for performance, showing first 100 pixels
        return {"total_pixels": len(pixels), "first_100_serpentine": pixels[:100]}
    finally:
        if os.path.exists(input_path): os.remove(input_path)

@app.post("/process-dct")
def apply_dct(file: UploadFile = File(...)):
    """
    Exercise 6: Uploads image, applies DCT, returns the visualized Transform.
    """
    input_path = save_upload(file)
    output_vis = f"dct_{file.filename}"
    output_rec = f"rec_{file.filename}"
    
    try:
        converter = DCT_Converter()
        converter.apply_dct(input_path, output_vis, output_rec)
        return FileResponse(output_vis, media_type="image/jpeg", filename="dct_visualization.jpg")
    finally:
        # Cleanup input, keep output for return (OS cleans temp files eventually or use background tasks)
        if os.path.exists(input_path): os.remove(input_path)

@app.post("/process-dwt")
def apply_dwt(file: UploadFile = File(...)):
    """
    Exercise 7: Uploads image, applies DWT (Haar), returns visualization.
    """
    input_path = save_upload(file)
    output_path = f"dwt_{file.filename}"
    dummy_rec = "dummy_rec.png"
    
    try:
        converter = DWT_Converter()
        converter.apply_dwt(input_path, output_path, dummy_rec)
        return FileResponse(output_path, media_type="image/jpeg", filename="dwt_visualization.jpg")
    finally:
        if os.path.exists(input_path): os.remove(input_path)

@app.post("/max-compression")
def max_compression(file: UploadFile = File(...)):
    """
    Exercise 5: Hard compression to Black and White.
    """
    input_path = save_upload(file)
    output_path = f"bw_{file.filename}"
    
    try:
        FFmpegAuto.max_compression(input_path, output_path)
        return FileResponse(output_path, media_type="image/jpeg", filename="bw_compression.jpg")
    finally:
        if os.path.exists(input_path): os.remove(input_path)
        
        
@app.post("/resize-image")
def resize_image(file: UploadFile = File(...), width: int = ..., height: int = ...):
    """
    Exercise 3: Resizes an image to specific dimensions.
    """
    input_path = save_upload(file)
    output_path = f"resized_{file.filename}"  # Changed name to be clear
    
    try:
        # FIX: Order must be (Input, Width, Height, Output)
        FFmpegAuto.resize_image(input_path, width, height, output_path)
        return FileResponse(output_path, media_type="image/jpeg", filename="resized_image.jpg")
    finally:
        if os.path.exists(input_path): os.remove(input_path)