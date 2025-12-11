from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/convert', methods=['POST'])
def convert():
    data = request.json
    filename = data.get('filename', 'input.mp4')
    
    cmd = f"ffmpeg -version" 
    
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        return jsonify({
            "message": f"FFMPEG processed {filename}",
            "output": result.stdout[:100] # Show first 100 chars of ffmpeg output
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)