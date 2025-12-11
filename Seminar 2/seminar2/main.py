import os
import shutil
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import List
import mimetypes
from fastapi import Query, BackgroundTasks
import subprocess

from s2_logic import ColorTranslator, DataSerializer, FFmpegAuto, DCT_Converter, DWT_Converter

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

# --- 2. DOCKER INTERACTION  ---
@app.post("/convert-video")
def convert_video(video_name: str):

    ffmpeg_url = "http://ffmpeg-service:5000/convert"
    try:
        response = requests.post(ffmpeg_url, json={"filename": video_name})
        return response.json()
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="FFMPEG Service unavailable")

# --- 3. SEMINAR 1 IMAGE LOGIC ---

@app.post("/serpentine-read")
def serpentine_scan(file: UploadFile = File(...)):

    input_path = save_upload(file)
    try:
        pixels = DataSerializer.serpentine_read(input_path)
        # Limit response size for performance, showing first 100 pixels
        return {"total_pixels": len(pixels), "first_100_serpentine": pixels[:100]}
    finally:
        if os.path.exists(input_path): os.remove(input_path)

@app.post("/process-dct")
def apply_dct(file: UploadFile = File(...)):

    input_path = save_upload(file)
    output_vis = f"dct_{file.filename}"
    output_rec = f"rec_{file.filename}"
    
    try:
        converter = DCT_Converter()
        converter.apply_dct(input_path, output_vis, output_rec)
        return FileResponse(output_vis, media_type="image/jpeg", filename="dct_visualization.jpg")
    finally:
        # Cleanup input, keep output for return 
        if os.path.exists(input_path): os.remove(input_path)

@app.post("/process-dwt")
def apply_dwt(file: UploadFile = File(...)):
 
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

    input_path = save_upload(file)
    output_path = f"bw_{file.filename}"
    
    try:
        FFmpegAuto.max_compression(input_path, output_path)
        return FileResponse(output_path, media_type="image/jpeg", filename="bw_compression.jpg")
    finally:
        if os.path.exists(input_path): os.remove(input_path)
        
        
@app.post("/resize")
def resize(file: UploadFile = File(...), width: int = ..., height: int = ..., isVideo: bool = ...):

    input_path = save_upload(file)
    output_path = f"resized_{file.filename}"  
    
    try:
        FFmpegAuto.resize(input_path, width, height, output_path)
        if (isVideo == True):
            return FileResponse(output_path, media_type="video/mp4", filename="resized_video.mp4")
        else:
            return FileResponse(output_path, media_type="image/jpeg", filename="resized_image.jpg")
    finally:
        if os.path.exists(input_path): os.remove(input_path)

def _safe_remove(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

@app.post("/set-chroma")
def set_chroma(file: UploadFile = File(...),
               subsampling: str = Query('4:2:0', description="Chroma subsampling: '4:2:0','4:2:2','4:4:4','4:0:0'"),
               background_tasks: BackgroundTasks = None):

    input_path = save_upload(file)
    safe_sub = subsampling.replace(':', '')
    output_path = f"chroma_{safe_sub}_{file.filename}"

    try:
        try:
            actual_output = FFmpegAuto.set_chroma_subsampling(input_path, output_path, subsampling)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            # bubble up ffmpeg stderr for debugging
            raise HTTPException(status_code=500, detail=str(e))

        # Guess media type for Content-Type header
        mime_type, _ = mimetypes.guess_type(actual_output)
        if mime_type is None:
            mime_type = "application/octet-stream"

        # Schedule the uploaded temp file for removal
        if background_tasks is not None:
            background_tasks.add_task(_safe_remove, input_path)
        else:
            # fallback immediate remove of input (keeps output)
            _safe_remove(input_path)

        return FileResponse(actual_output, media_type=mime_type, filename=os.path.basename(actual_output))
    finally:
        if background_tasks is None and os.path.exists(input_path):
            _safe_remove(input_path)
            
@app.post("/relevant_information")
def relevant_information(file: UploadFile = File(...)):

    input_path = save_upload(file)
    try:
        try:
            info_lines = DataSerializer.inportant_information(input_path)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to extract information: {e}")

        # Join into a single string for the response
        joined = "\n".join(info_lines)
        return PlainTextResponse(joined)
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

@app.post("/create_bbb_container")
def create_bbb_container(file: UploadFile = File(...), duration: int = 20):

    input_path = save_upload(file)
    base_name = os.path.splitext(file.filename)[0]
    output_path = f"bbb_{base_name}_{duration}s.mp4"

    try:
        try:
            out = FFmpegAuto.create_bbb_container(input_path, output_path, duration=duration)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

        return FileResponse(out, media_type="video/mp4", filename=os.path.basename(out))
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

@app.post("/visualize_motion_vectors")
def visualize_motion_vectors(file: UploadFile = File(...)):

    input_path = save_upload(file)
    base_name = os.path.splitext(file.filename)[0]
    output_path = f"mv_{base_name}.mp4"

    try:
        try:
            out = FFmpegAuto.visualize_motion_vectors(input_path, output_path)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

        return FileResponse(out, media_type="video/mp4", filename=os.path.basename(out))
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

@app.post("/yuv_histogram")
def yuv_histogram(file: UploadFile = File(...)):

    input_path = save_upload(file)
    base_name = os.path.splitext(file.filename)[0]
    output_path = f"yuv_hist_{base_name}.mp4"

    try:
        try:
            out = FFmpegAuto.yuv_histogram(input_path, output_path)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

        return FileResponse(out, media_type="video/mp4", filename=os.path.basename(out))
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

@app.post("/count-tracks")
def count_tracks(file: UploadFile = File(...)):

    input_path = save_upload(file)
    try:
        try:
            num, streams = FFmpegAuto.probe_tracks(input_path)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

        msg = f"The file contains {num} track{'s' if num != 1 else ''}."

        return JSONResponse({
            "message": msg,
            "filename": file.filename,
            "total_tracks": num,
            "streams": streams
        })

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
