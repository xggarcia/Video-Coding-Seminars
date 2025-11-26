import pytest
import os
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

# Import the module we just created
from s1_logic import ColorTranslator, DataSerializer, FFmpegAuto

# --- Test ColorTranslator ---
def test_rgb_to_yuv():
    # Test White (255, 255, 255)
    y, u, v = ColorTranslator.rgb_to_yuv(255, 255, 255)
    # Expected approx: Y=255, U=0, V=0 (depending on precision)
    # Using the formula in code:
    # Y = 0.299*255 + 0.587*255 + 0.114*255 = 255.0
    assert int(y) == 255

def test_yuv_to_rgb():
    # Test conversion back
    r, g, b = ColorTranslator.rgb_to_yuv(255, 0, 0, mode='YUV_to_RGB')
    # Should get White back
    assert r == 255 and g == 255 and b == 255

# --- Test DataSerializer ---
def test_run_length_encoding():
    data = ['A', 'A', 'B', 'C', 'C', 'C']
    expected = ['A', 2, 'B', 1, 'C', 3]
    assert DataSerializer.run_length_encoding(data) == expected

def test_run_length_encoding_empty():
    assert DataSerializer.run_length_encoding([]) == []

def test_serpentine_read(tmp_path):
    # 1. Create a tiny 2x2 temporary image for testing
    # (0,0) (1,0)
    # (0,1) (1,1)
    # Serpentine 2x2: (0,0) -> (1,0) -> (0,1) -> (1,1)
    
    img = Image.new('RGB', (2, 2))
    pixels = img.load()
    pixels[0,0] = (10, 10, 10)
    pixels[1,0] = (20, 20, 20)
    pixels[0,1] = (30, 30, 30)
    pixels[1,1] = (40, 40, 40)
    
    p = tmp_path / "test_img.png"
    img.save(p)
    
    result = DataSerializer.serpentine_read(str(p))
    
    # Check sequence
    assert result[0] == (10, 10, 10) # 0,0
    assert result[1] == (20, 20, 20) # 1,0
    assert result[2] == (30, 30, 30) # 0,1
    assert result[3] == (40, 40, 40) # 1,1

# --- Test FFmpegAuto (Mocking Subprocess) ---
@patch('subprocess.run')
def test_resize_image(mock_run):
    FFmpegAuto.resize_image("input.jpg", 100, 100, "output.jpg")
    
    # Assert subprocess was called
    mock_run.assert_called_once()
    
    # Check if arguments were correct
    args = mock_run.call_args[0][0]
    assert 'ffmpeg' in args
    assert 'scale=100:100' in args

@patch('subprocess.run')
def test_max_compression(mock_run):
    # This method calls other methods, so we expect multiple subprocess calls
    FFmpegAuto.max_compression("input.jpg", "output.jpg")
    
    assert mock_run.call_count >= 2 # Resize + Quantize