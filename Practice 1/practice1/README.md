Markdown

# P1 - API & Dockerization

## 📋 Project Overview
This project implements a multimedia processing API using **FastAPI** and **Docker**. It integrates algorithms from Seminar 1 (Color spaces, RLE, DCT, DWT) and orchestrates a microservice architecture where heavy video processing is delegated to a separate **FFMPEG container**.

This practice is part of the **Video Coding Seminars** repository.

---

## 🏗️ Architecture
The solution uses **Docker Compose** to orchestrate two containers:

1.  **Main API (`practice1-api`)**:
    * Built with **FastAPI**.
    * Handles image processing logic (OpenCV, PIL, NumPy).
    * Contains the logic adapted from Seminar 1.
    * Exposes endpoints for user interaction.

2.  **FFMPEG Service (`practice1-ffmpeg-service`)**:
    * A lightweight **Flask** microservice.
    * Dedicated solely to executing FFMPEG commands.
    * Receives HTTP requests from the Main API to perform video conversions.

**Communication:** The Main API sends HTTP POST requests to the FFMPEG Service within the internal Docker network.

---

## 🚀 How to Run
Since this project is part of a monorepo, please follow these specific steps to navigate to the correct directory before running the containers.

### 1. Clone the Repository
Clone the main repository containing all seminars and practices:
```bash
git clone [https://github.com/](https://github.com/)xggarcia/Video-Coding-Seminars.git

### 2. Navigate to the Practice Folder
Important: You must move into the specific subfolder for Practice 1 before running Docker.

You can use the following line of code in cmd:

--> cd "Video-Coding-Seminars/Practice 1/practice1"

### 3. Start the Application
Run the following command to build and launch both containers:

--> docker compose up --build

### 4. Access the API
Once the logs show the service is running:

You can go to the following url in your browser: http://localhost:8000/docs


#-------------------------------------------------------------------------------


### 1. General & Seminar 1 Logic

GET /rgb-to-yuv: Converts RGB color values to YUV space.

POST /encode-rle: Performs Run-Length Encoding (RLE) on a list of data.

### 2. Image Processing (OpenCV & S1)
These endpoints accept image file uploads (multipart/form-data):

POST /serpentine-read: Reads pixels in a zigzag pattern. Returns a list of the pixels.

POST /process-dct: Applies Discrete Cosine Transform (DCT) and returns the visualization (can be downloaded if needed).

POST /process-dwt: Applies Discrete Wavelet Transform (DWT - Haar) and returns the visualitzation (can be downloaded if needed). 

POST /resize-image: Resizes an uploaded image to specified dimensions (can be downloaded if needed).

POST /max_compresion: Resizes the image to 160 x 120 and puts the image in b/w with just 8 colors. 

3. Video Processing (Container Interaction)
POST /convert-video:

Input: JSON body {"video_name": "example.mp4"}

Action: Triggers the ffmpeg-service container to process the video.

Output: Returns execution logs from the secondary container.

👥 Authors
Student Name: Guillem García 

Student Name: Jofre Geli

University: Universitat Pompeu Fabra (UPF)