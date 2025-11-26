# ffmpeg_service/Dockerfile
FROM python:3.9-slim

# Install FFMPEG 
RUN apt-get update && apt-get install -y ffmpeg

# Install Flask for the microservice
RUN pip install flask

WORKDIR /app
COPY worker.py .

CMD ["python", "worker.py"]