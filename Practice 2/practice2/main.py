import os
import shutil
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import mimetypes
from fastapi import Query, BackgroundTasks
import subprocess


# Import your logic
from p2_logic import ColorTranslator, DataSerializer, FFmpegAuto, DCT_Converter, DWT_Converter

app = FastAPI(title="Multimedia API", description="API for Practice 2 & Dockerization")

# Mount static files directory for the GUI
app.mount("/static", StaticFiles(directory="static"), name="static")

# Helper to save uploaded files temporarily
def save_upload(upload_file: UploadFile) -> str:
    path = f"temp_{upload_file.filename}"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return path

# Helper to cleanup files and folders
def cleanup_path(path: str):
    try:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    except Exception as e:
        print(f"Cleanup warning: {e}")

# --- 1. BASIC ENDPOINTS ---
@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serve the GUI homepage"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api")
def api_info():
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

    ffmpeg_url = "http://ffmpeg-service:5000/convert"
    try:
        response = requests.post(ffmpeg_url, json={"filename": video_name})
        return response.json()
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="FFMPEG Service unavailable")

# --- 3. PRACTICE 2 IMAGE LOGIC ---

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
        response = FileResponse(output_vis, media_type="image/jpeg", filename="dct_visualization.jpg")
        # Schedule cleanup after response is sent
        response.background = BackgroundTasks()
        response.background.add_task(cleanup_path, output_vis)
        response.background.add_task(cleanup_path, output_rec)
        return response
    finally:
        cleanup_path(input_path)

@app.post("/process-dwt")
def apply_dwt(file: UploadFile = File(...)):

    input_path = save_upload(file)
    output_path = f"dwt_{file.filename}"
    dummy_rec = "dummy_rec.png"
    
    try:
        converter = DWT_Converter()
        converter.apply_dwt(input_path, output_path, dummy_rec)
        response = FileResponse(output_path, media_type="image/jpeg", filename="dwt_visualization.jpg")
        # Schedule cleanup after response is sent
        response.background = BackgroundTasks()
        response.background.add_task(cleanup_path, output_path)
        response.background.add_task(cleanup_path, dummy_rec)
        return response
    finally:
        cleanup_path(input_path)

@app.post("/max-compression")
def max_compression(file: UploadFile = File(...)):

    input_path = save_upload(file)
    output_path = f"bw_{file.filename}"
    
    try:
        FFmpegAuto.max_compression(input_path, output_path)
        response = FileResponse(output_path, media_type="image/jpeg", filename="bw_compression.jpg")
        # Schedule cleanup after response is sent
        response.background = BackgroundTasks()
        response.background.add_task(cleanup_path, output_path)
        return response
    finally:
        cleanup_path(input_path)
        
        
@app.post("/resize")
def resize(file: UploadFile = File(...), width: int = ..., height: int = ..., isVideo: bool = ...):

    input_path = save_upload(file)
    output_path = f"resized_{file.filename}"  # Changed name to be clear
    
    try:
        # FIX: Order must be (Input, Width, Height, Output)
        FFmpegAuto.resize(input_path, width, height, output_path)
        if isVideo:
            media_type = "video/mp4"
            filename = "resized_video.mp4"
        else:
            media_type = "image/jpeg"
            filename = "resized_image.jpg"
        response = FileResponse(output_path, media_type=media_type, filename=filename)
        # Schedule cleanup after response is sent
        response.background = BackgroundTasks()
        response.background.add_task(cleanup_path, output_path)
        return response
    finally:
        cleanup_path(input_path)

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
    # start with original filename, converter may change extension (e.g., to .mkv)
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

        response = FileResponse(actual_output, media_type=mime_type, filename=os.path.basename(actual_output))
        # Schedule cleanup after response is sent
        response.background = BackgroundTasks()
        response.background.add_task(cleanup_path, actual_output)
        return response
    finally:
        cleanup_path(input_path)
            
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

        response = FileResponse(out, media_type="video/mp4", filename=os.path.basename(out))
        # Schedule cleanup after response is sent
        response.background = BackgroundTasks()
        response.background.add_task(cleanup_path, out)
        return response
    finally:
        cleanup_path(input_path)

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

        response = FileResponse(out, media_type="video/mp4", filename=os.path.basename(out))
        # Schedule cleanup after response is sent
        response.background = BackgroundTasks()
        response.background.add_task(cleanup_path, out)
        return response
    finally:
        cleanup_path(input_path)

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

        response = FileResponse(out, media_type="video/mp4", filename=os.path.basename(out))
        # Schedule cleanup after response is sent
        response.background = BackgroundTasks()
        response.background.add_task(cleanup_path, out)
        return response
    finally:
        cleanup_path(input_path)

@app.post("/count_tracks")
def count_tracks(file: UploadFile = File(...)):

    input_path = save_upload(file)
    try:
        try:
            num, streams = FFmpegAuto.probe_tracks(input_path)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Human-friendly summary message
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

@app.post("/convert_codec")
def convert_codec(file: UploadFile = File(...), codec: str = Query(..., description="Target codec: vp8, vp9, h265, or av1")):

    input_path = save_upload(file)
    output_dir = "converted_videos"
    
    try:
        try:
            output_path = FFmpegAuto.convert_video_codec(input_path, output_dir, codec)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
        
        # Determine media type based on file extension
        if output_path.endswith('.webm'):
            media_type = "video/webm"
        elif output_path.endswith('.mp4'):
            media_type = "video/mp4"
        elif output_path.endswith('.mkv'):
            media_type = "video/x-matroska"
        else:
            media_type = "video/mp4"
        
        response = FileResponse(output_path, media_type=media_type, filename=os.path.basename(output_path))
        # Schedule cleanup after response is sent
        response.background = BackgroundTasks()
        response.background.add_task(cleanup_path, output_path)
        response.background.add_task(cleanup_path, output_dir)
        return response
    finally:
        cleanup_path(input_path)

@app.post("/create_encoding_ladder")
def create_encoding_ladder(file: UploadFile = File(...), codec: str = Query('h265', description="Target codec: h264, h265, vp9, or av1")):

    input_path = save_upload(file)
    output_dir = "encoding_ladder"
    
    try:
        try:
            results = FFmpegAuto.create_encoding_ladder(input_path, output_dir, codec)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
        
        response = JSONResponse({
            "message": f"Successfully created encoding ladder with {len(results)} variants",
            "codec": codec,
            "variants": results
        })
        # Schedule cleanup after response is sent
        response.background = BackgroundTasks()
        response.background.add_task(cleanup_path, output_dir)
        return response
    finally:
        cleanup_path(input_path)
