import os
import subprocess
import numpy as np
import cv2
import pywt
from PIL import Image
from os import path, remove

# --- EXERCISE 2: COLOR TRANSLATION ---
class ColorTranslator:
    """
    Handles mathematical color space conversions.
    """
    @staticmethod
    def rgb_to_yuv(v1, v2, v3, mode='RGB_to_YUV'):
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


# --- EXERCISE 4 & 5: DATA ENCODING/TRAVERSAL ---
class DataSerializer:
    """
    Handles pixel reading patterns and basic compression algorithms (RLE).
    """
    @staticmethod
    def serpentine_read(file_path):
        """
        Exercise 4: Reads image pixels in a zigzag (serpentine) pattern.
        This simulates the way JPEG reads 8x8 blocks.
        """
        if not path.exists(file_path):
            print(f"File not found: {file_path}")
            return []

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
    def run_length_encoding(array):
        """
        Exercise 5 (Part 2): RLE Encoding.
        Input: [A, A, B] -> Output: [A, 2, B, 1]
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


# --- EXERCISE 3 & 5: FFMPEG AUTOMATION ---
class FFmpegAuto:
    """
    Wrapper for FFmpeg subprocess calls to automate resizing and compression.
    """
    @staticmethod
    def resize_image(input_path, new_width, new_height, output_path):
        """
        Exercise 3: Resize image using ffmpeg.
        """
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-vf', f"scale={new_width}:{new_height}",
            output_path
        ]
        
        subprocess.run(cmd)
        

    @staticmethod
    def to_black_and_white(file_path, output_path):
        """
        Exercise 5 (Part 1): Hard compression to 1-bit Black and White.
        """
        if path.exists(output_path): remove(output_path)
        
        # Pixels > 127 become 255 (White), others become 0 (Black)
        cmd = [
            'ffmpeg', '-y', '-i', file_path,
            '-vf', "format=gray,geq=lum='if(gt(lum,127),255,0)'",
            output_path
        ]
        subprocess.run(cmd)


    @staticmethod
    def quantize_grayscale(input_path, output_path, num_colors=8):
        """
        Exercise 5 (Part 1 continued): Grayscale quantization with specific color palette.
        """
        if path.exists(output_path): remove(output_path)
            
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
        subprocess.run(cmd)
        
    @staticmethod
    def max_compression(input_path, output_path):

            FFmpegAuto.resize_image(input_path, 160, 120, "temp_resized.jpg")
            FFmpegAuto.quantize_grayscale("temp_resized.jpg", output_path)
            remove("temp_resized.jpg")



# --- EXERCISE 6: DCT CLASS ---
class DCT_Converter:
    """
    Handles Discrete Cosine Transform operations (JPEG core).
    """
    def __init__(self, block_size=8):
        self.block_size = block_size

    def apply_dct(self, input_path, output_dct, output_idct):
        if not path.exists(input_path): return
        
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
                
                # DCT
                dst = cv2.dct(block)
                dct_view[i:i+self.block_size, j:j+self.block_size] = dst
                
                # IDCT
                src = cv2.idct(dst)
                reconstructed[i:i+self.block_size, j:j+self.block_size] = src

        # Save Visualizations
        dct_log = np.log(np.abs(dct_view) + 1)
        dct_norm = cv2.normalize(dct_log, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(output_dct, np.uint8(dct_norm))
        cv2.imwrite(output_idct, np.uint8(reconstructed))
        print(f"✅ DCT processed: {output_dct}")


# --- EXERCISE 7: DWT CLASS ---
class DWT_Converter:
    """
    Handles Discrete Wavelet Transform operations (JPEG2000 core).
    """
    def __init__(self, wavelet='haar'):
        self.wavelet = wavelet

    def apply_dwt(self, input_path, output_dwt, output_idwt):
        if not path.exists(input_path): return
        
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure even dimensions
        h, w = img.shape
        img = img[0:(h // 2) * 2, 0:(w // 2) * 2]
        
        # DWT
        coeffs = pywt.dwt2(img, self.wavelet)
        LL, (LH, HL, HH) = coeffs
        
        # Visualization
        def norm(arr):
            return cv2.normalize(np.abs(arr), None, 0, 255, cv2.NORM_MINMAX)
            
        vis = np.vstack([
            np.hstack([norm(LL), norm(HL)]),
            np.hstack([norm(LH), norm(HH)])
        ])
        cv2.imwrite(output_dwt, np.uint8(vis))
        
        # IDCT
        rec = pywt.idwt2(coeffs, self.wavelet)
        cv2.imwrite(output_idwt, np.uint8(np.clip(rec, 0, 255)))
        print(f"✅ DWT processed: {output_dwt}")


# --- DEMONSTRATION (Exercise 8 part A) ---
if __name__ == "__main__":
    print("\n--- STARTING DEMONSTRATION OF ALL EXERCISES ---\n")
    
    # 0. Setup Paths
    base_dir = "Seminar 1"
    if not path.exists(base_dir): os.makedirs(base_dir)
    
    # Create a dummy input image if it doesn't exist
    input_jpg = path.join(base_dir, "input.jpg")
    if not path.exists(input_jpg):
        # Create a gradient image for testing
        grad = np.linspace(0, 255, 256*256).reshape(256, 256).astype(np.uint8)
        cv2.imwrite(input_jpg, grad)
        print("Created dummy input.jpg")

    # EXERCISE 2: COLOR
    print("[Ex 2] Translating White (255,255,255)...")
    y, u, v = ColorTranslator.rgb_to_yuv(255, 255, 255)
    print(f"   Result: Y={y}, U={u}, V={v}")
    
    # EXERCISE 3 & 5: FFMPEG
    print("\n[Ex 3 & 5] FFmpeg compressions...")
    out_bw = path.join(base_dir, "output_bw.png")
    out_gray = path.join(base_dir, "output_gray.png")
    
    # Resize first
    resize = path.join(base_dir, "resize.jpg")
    FFmpegAuto.resize_image(input_jpg, 320/6, 240/6, resize)
    
    # Apply effects
    black_and_white = path.join(base_dir, "output_bw.png")
    gray_quantized = path.join(base_dir, "output_gray.png")
    FFmpegAuto.quantize_grayscale(input_jpg, gray_quantized, num_colors=20)
    
    
    max_compression = path.join(base_dir, "output_max_compression.png")
    FFmpegAuto.max_compression(input_jpg, max_compression)
    
    # EXERCISE 4: SERPENTINE
    print("\n[Ex 4] Serpentine Scan...")
    pixels = DataSerializer.serpentine_read(input_jpg)
    print(f"   Scanned {len(pixels)} pixels.")
    
    # EXERCISE 5: RLE
    print("\n[Ex 5] Run Length Encoding...")
    data = ['A', 'A', 'A', 'B', 'C', 'C']
    encoded = DataSerializer.run_length_encoding(data)
    print(f"   Input: {data}")
    print(f"   Encoded: {encoded}")
    
    # EXERCISE 6: DCT
    print("\n[Ex 6] Discrete Cosine Transform...")
    dct = DCT_Converter()
    dct.apply_dct(input_jpg, path.join(base_dir, "dct_vis.png"), path.join(base_dir, "dct_rec.png"))
    
    # EXERCISE 7: DWT
    print("\n[Ex 7] Discrete Wavelet Transform...")
    dwt = DWT_Converter()
    dwt.apply_dwt(input_jpg, path.join(base_dir, "dwt_vis.png"), path.join(base_dir, "dwt_rec.png"))
    
    print("\n--- DEMONSTRATION COMPLETE ---")