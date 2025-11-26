import os
import subprocess
import numpy as np
import cv2
import pywt
from PIL import Image
from typing import List, Tuple, Union

class ColorTranslator:
    """
    Handles mathematical color space conversions.
    """
    @staticmethod
    def rgb_to_yuv(v1: int, v2: int, v3: int, mode: str = 'RGB_to_YUV') -> Tuple[float, float, float]:
        """
        Translates between RGB and YUV values.
        """
        if mode == 'YUV_to_RGB':
            Y, U, V = v1, v2, v3
            # Integer conversion with clamping to 0-255
            R = int(np.clip(Y + 1.13983 * V, 0, 255))
            G = int(np.clip(Y - 0.39465 * U - 0.58060 * V, 0, 255))
            B = int(np.clip(Y + 2.03211 * U, 0, 255))
            return (R, G, B)
        else:  # Default: RGB_to_YUV
            R, G, B = v1, v2, v3
            Y = round(0.299 * R + 0.587 * G + 0.114 * B, 2)
            U = round(-0.14713 * R - 0.28886 * G + 0.436 * B, 2)
            V = round(0.615 * R - 0.51499 * G - 0.10001 * B, 2)
            return (Y, U, V)

class DataSerializer:
    """
    Handles pixel reading patterns and basic compression algorithms (RLE).
    """
    @staticmethod
    def serpentine_read(file_path: str) -> List[Tuple[int, int, int]]:
        """
        Reads image pixels in a zigzag (serpentine) pattern.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        img = Image.open(file_path).convert("RGB")
        width, height = img.size
        pixels = img.load()

        serpentine_pixels = []

        for d in range(width + height - 1):
            if d % 2 == 0: # Moving Up-Right
                y = min(d, height - 1)
                x = d - y
                while y >= 0 and x < width:
                    serpentine_pixels.append(pixels[x, y])
                    x += 1
                    y -= 1
            else: # Moving Down-Left
                x = min(d, width - 1)
                y = d - x
                while x >= 0 and y < height:
                    serpentine_pixels.append(pixels[x, y])
                    x -= 1
                    y += 1
        return serpentine_pixels

    @staticmethod
    def run_length_encoding(array: List) -> List:
        """
        RLE Encoding. Input: [A, A, B] -> Output: [A, 2, B, 1]
        """
        if not array: return []
        
        output = []
        counter = 1
        for i in range(len(array) - 1):
            if array[i] == array[i+1]:
                counter += 1
            else:
                output.append(array[i])
                output.append(counter)
                counter = 1
        
        # Append the last sequence
        output.append(array[-1])
        output.append(counter)
        return output

class FFmpegAuto:
    """
    Wrapper for FFmpeg subprocess calls.
    """
    @staticmethod
    def resize_image(input_path: str, new_width: int, new_height: int, output_path: str):
        # Ensure dimensions are integers
        w, h = int(new_width), int(new_height)
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-vf', f"scale={w}:{h}",
            output_path
        ]
        subprocess.run(cmd, check=True) # check=True raises error on failure

    @staticmethod
    def to_black_and_white(file_path: str, output_path: str):
        if os.path.exists(output_path): os.remove(output_path)
        cmd = [
            'ffmpeg', '-y', '-i', file_path,
            '-vf', "format=gray,geq=lum='if(gt(lum,127),255,0)'",
            output_path
        ]
        subprocess.run(cmd, check=True)

    @staticmethod
    def quantize_grayscale(input_path: str, output_path: str, num_colors: int = 8):
        if os.path.exists(output_path): os.remove(output_path)
        filter_cmd = (
            f"[0:v]format=gray,split[s0][s1];"
            f"[s0]palettegen=max_colors={num_colors}[p];"
            f"[s1][p]paletteuse"
        )
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-filter_complex', filter_cmd,
            output_path
        ]
        subprocess.run(cmd, check=True)
        
    @staticmethod
    def max_compression(input_path: str, output_path: str):
        temp_resized = "temp_resized.jpg"
        try:
            FFmpegAuto.resize_image(input_path, 160, 120, temp_resized)
            FFmpegAuto.quantize_grayscale(temp_resized, output_path)
        finally:
            if os.path.exists(temp_resized):
                os.remove(temp_resized)

class DCT_Converter:
    """
    Handles Discrete Cosine Transform operations.
    """
    def __init__(self, block_size: int = 8):
        self.block_size = block_size

    def apply_dct(self, input_path: str, output_dct: str, output_idct: str):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")
        
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        
        # Resize to fit block size
        h_new = int(np.ceil(h / self.block_size) * self.block_size)
        w_new = int(np.ceil(w / self.block_size) * self.block_size)
        if h != h_new or w != w_new:
            img = cv2.resize(img, (w_new, h_new))

        img_float = np.float32(img)
        dct_view = np.zeros_like(img_float)
        reconstructed = np.zeros_like(img_float)

        for i in range(0, h_new, self.block_size):
            for j in range(0, w_new, self.block_size):
                block = img_float[i:i+self.block_size, j:j+self.block_size]
                dst = cv2.dct(block)
                dct_view[i:i+self.block_size, j:j+self.block_size] = dst
                src = cv2.idct(dst)
                reconstructed[i:i+self.block_size, j:j+self.block_size] = src

        dct_log = np.log(np.abs(dct_view) + 1)
        dct_norm = cv2.normalize(dct_log, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(output_dct, np.uint8(dct_norm))
        cv2.imwrite(output_idct, np.uint8(reconstructed))

class DWT_Converter:
    """
    Handles Discrete Wavelet Transform operations.
    """
    def __init__(self, wavelet: str = 'haar'):
        self.wavelet = wavelet

    def apply_dwt(self, input_path: str, output_dwt: str, output_idwt: str):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")
        
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        img = img[0:(h // 2) * 2, 0:(w // 2) * 2]
        
        coeffs = pywt.dwt2(img, self.wavelet)
        LL, (LH, HL, HH) = coeffs
        
        def norm(arr):
            return cv2.normalize(np.abs(arr), None, 0, 255, cv2.NORM_MINMAX)
            
        vis = np.vstack([
            np.hstack([norm(LL), norm(HL)]),
            np.hstack([norm(LH), norm(HH)])
        ])
        cv2.imwrite(output_dwt, np.uint8(vis))
        
        rec = pywt.idwt2(coeffs, self.wavelet)
        cv2.imwrite(output_idwt, np.uint8(np.clip(rec, 0, 255)))