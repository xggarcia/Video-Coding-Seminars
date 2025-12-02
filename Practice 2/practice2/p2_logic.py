import os
import subprocess
import json
import subprocess
import numpy as np
import cv2
import pywt
from PIL import Image
from typing import List, Tuple, Union, Dict, Optional

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
    def inportant_information(file_path: str):
        """Extract important video metadata using ffprobe.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        cmd = ['ffprobe', '-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', file_path]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(proc.stdout)
        except Exception:
            return ["Error: ffprobe failed or not available"]

        fmt = info.get('format', {})
        streams = info.get('streams', [])
        v = next((s for s in streams if s.get('codec_type') == 'video'), None)

        out: List[str] = []
        # duration (seconds)
        if fmt.get('duration'):
            try:
                out.append(f"Duration: {float(fmt['duration']):.3f} s")
            except Exception:
                pass
        # size (bytes)
        if fmt.get('size'):
            try:
                out.append(f"Size: {int(fmt['size'])} bytes")
            except Exception:
                pass

        if v:
            w, h = v.get('width'), v.get('height')
            if w and h:
                out.append(f"Resolution: {w}x{h}")

            pix = v.get('pix_fmt')
            subs = {'yuv420p': '4:2:0', 'yuv422p': '4:2:2', 'yuv444p': '4:4:4'}.get(pix, 'unknown') if pix else 'unknown'
            out.append(f"Chroma subsampling: {subs}")

            if v.get('codec_name'):
                out.append(f"Codec: {v.get('codec_name')}")

            fr = v.get('avg_frame_rate') or v.get('r_frame_rate')
            if fr and fr != '0/0':
                try:
                    if '/' in fr:
                        n, d = fr.split('/')
                        fps = float(n) / float(d)
                    else:
                        fps = float(fr)
                    out.append(f"Frame rate: {fps:.2f} fps")
                except Exception:
                    pass

        if not out:
            out.append("No information available")

        return out

    
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
    def resize(input_path: str, new_width: int, new_height: int, output_path: str):
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
            # use the existing `resize` staticmethod
            FFmpegAuto.resize(input_path, 160, 120, temp_resized)
            FFmpegAuto.quantize_grayscale(temp_resized, output_path)
        finally:
            if os.path.exists(temp_resized):
                os.remove(temp_resized)

    @staticmethod
    def set_chroma_subsampling(input_path: str, output_path: str, subsampling: str = '4:2:0') -> str:
        """
        Convert chroma subsampling / pixel format for images or videos.
        Returns the actual output_path used (may change extension for certain codecs).
        Raises ValueError on bad subsampling, RuntimeError if ffmpeg fails.
        """
        mapping = {
            '4:4:4': 'yuv444p',
            '4:2:2': 'yuv422p',
            '4:2:0': 'yuv420p',
            '4:0:0': 'gray'
        }
        if subsampling not in mapping:
            raise ValueError(f"Unsupported subsampling '{subsampling}'. Supported: {', '.join(mapping.keys())}")

        pix_fmt = mapping[subsampling]

        # Remove existing output if present
        if os.path.exists(output_path):
            os.remove(output_path)

        # Non-monochrome path: use libx264
        if pix_fmt != "gray":
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-vf', f"format={pix_fmt}",
                '-pix_fmt', pix_fmt,
                '-c:v', 'libx264',
                output_path
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}): {proc.stderr.strip()}")
            return output_path

        # Monochrome path: try ffv1 (compressed, in mkv). If it fails, fallback to rawvideo (.avi)
        base, _ = os.path.splitext(output_path)
        mkv_out = base + ".mkv"
        cmd_ffv1 = [
            'ffmpeg', '-y', '-i', input_path,
            '-vf', f"format={pix_fmt}",
            '-pix_fmt', pix_fmt,
            '-c:v', 'ffv1',
            mkv_out
        ]

        proc = subprocess.run(cmd_ffv1, capture_output=True, text=True)
        if proc.returncode == 0:
            return mkv_out

        # ffv1 failed: fallback to rawvideo (guaranteed)
        avi_out = base + ".avi"
        cmd_raw = [
            'ffmpeg', '-y', '-i', input_path,
            '-vf', f"format={pix_fmt}",
            '-pix_fmt', pix_fmt,
            '-c:v', 'rawvideo',
            avi_out
        ]

        proc2 = subprocess.run(cmd_raw, capture_output=True, text=True)
        if proc2.returncode != 0:
            combined = f"ffv1 stderr: {proc.stderr.strip()}\nrawvideo stderr: {proc2.stderr.strip()}"
            raise RuntimeError(f"ffmpeg failed for both ffv1 and rawvideo. Details:\n{combined}")

        return avi_out

    @staticmethod
    def create_bbb_container(input_path: str, output_path: str, duration: int = 20) -> str:
        """
        Create a new MP4 container from `input_path` trimmed to `duration` seconds.
        - Cuts video to 20 seconds (or specified duration)
        - Exports AAC mono track
        - Exports MP3 stereo with lower bitrate
        - Exports AC3 codec track
        - Packages everything into a single .mp4
        
        Returns the path to the final MP4 output.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")

        # Use MP4 as required
        base, _ = os.path.splitext(output_path)
        final_output = base + '.mp4'

        # Check if input has audio stream
        probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 
                     'stream=codec_type', '-of', 'default=noprint_wrappers=1:nokey=1', input_path]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        has_audio = probe_result.stdout.strip() == 'audio'

        if has_audio:
            # Input has audio: map ONLY the first video stream (exclude mjpeg thumbnails) and create audio tracks
            cmd = [
                'ffmpeg', '-y', '-i', input_path, '-t', str(duration),
                # Map only the first video stream (0:v:0) to exclude mjpeg/thumbnail streams
                '-map', '0:v:0',
                # Audio track 1: AAC mono
                '-map', '0:a:0', '-c:a:0', 'aac', '-ac:a:0', '1', '-b:a:0', '128k',
                # Audio track 2: MP3 stereo lower bitrate (MP4 supports MP3)
                '-map', '0:a:0', '-c:a:1', 'libmp3lame', '-ac:a:1', '2', '-b:a:1', '96k',
                # Audio track 3: AC3 stereo (MP4 supports AC3 via mov_text)
                '-map', '0:a:0', '-c:a:2', 'ac3', '-ac:a:2', '2', '-b:a:2', '192k',
                '-c:v', 'copy',
                # Use movflags to ensure MP4 compatibility
                '-movflags', '+faststart',
                final_output
            ]
        else:
            # No audio: generate silent audio tracks
            cmd = [
                'ffmpeg', '-y', '-i', input_path, '-f', 'lavfi', '-i', f'anullsrc=r=44100:cl=mono', 
                '-t', str(duration),
                # Map only first video stream to exclude thumbnails
                '-map', '0:v:0',
                # Audio track 1: AAC mono (silent)
                '-map', '1:a', '-c:a:0', 'aac', '-ac:a:0', '1', '-b:a:0', '128k',
                # Audio track 2: MP3 stereo (silent)
                '-map', '1:a', '-c:a:1', 'libmp3lame', '-ac:a:1', '2', '-b:a:1', '96k',
                # Audio track 3: AC3 stereo (silent)
                '-map', '1:a', '-c:a:2', 'ac3', '-ac:a:2', '2', '-b:a:2', '192k',
                '-c:v', 'copy',
                '-movflags', '+faststart',
                '-shortest',
                final_output
            ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr}")

        return final_output

    @staticmethod
    def visualize_motion_vectors(input_path: str, output_path: str) -> str:
        """
        Visualize macroblocks and motion vectors in a video.
        Uses ffmpeg codecview filter to display motion vectors overlaid on the video.
        
        Returns the path to the output video with motion vector visualization.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")

        # Ensure output is .mp4
        base, _ = os.path.splitext(output_path)
        final_output = base + '.mp4'

        # Use ffmpeg codecview filter to visualize motion vectors
        # mv=pf shows forward predicted motion vectors
        # mv=bf shows backward predicted motion vectors  
        # mv=pf+bf shows both
        cmd = [
            'ffmpeg', '-y',
            # Decode with motion vector extraction flags
            '-flags2', '+export_mvs',
            '-i', input_path,
            # Apply codecview filter to visualize motion vectors
            '-vf', 'codecview=mv=pf+bf',
            # Re-encode to ensure visualization is embedded
            '-c:v', 'libx264',
            '-crf', '18',
            # Copy audio if present
            '-c:a', 'copy',
            final_output
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg motion vector visualization failed: {proc.stderr}")

        return final_output

    @staticmethod
    def yuv_histogram(input_path: str, output_path: str) -> str:
        """
        Create a video with YUV histogram overlay.
        Shows the distribution of Y (luma), U (Cb), and V (Cr) components
        as histograms overlaid on the video.
        
        Returns the path to the output video with YUV histogram visualization.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")

        # Ensure output is .mp4
        base, _ = os.path.splitext(output_path)
        final_output = base + '.mp4'

        # Use ffmpeg histogram filter for YUV components
        # Map only first video stream to exclude mjpeg thumbnails
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            # Map only the first video stream (exclude thumbnails)
            '-map', '0:v:0',
            # Apply histogram filter for YUV components (components=7 = Y+U+V)
            '-vf', 'split=2[a][b],[b]histogram=level_height=200:display_mode=overlay:levels_mode=linear:components=7[hh],[a][hh]overlay',
            # Re-encode
            '-c:v', 'libx264',
            '-crf', '18',
            # Copy audio if present
            '-map', '0:a?',
            '-c:a', 'copy',
            final_output
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg YUV histogram visualization failed: {proc.stderr}")

        return final_output

    @staticmethod
    def probe_tracks(file_path: str) -> Tuple[int, List[Dict]]:
        """
        Uses ffprobe to inspect an MP4/multi-stream container and returns:
            (number_of_streams, [stream_info_dicts...])

        stream_info_dicts include: index, codec_type, codec_name, codec_long_name,
        width, height, sample_rate, channels, duration (when available).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # ffprobe command to list streams in JSON
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=index,codec_type,codec_name,codec_long_name,width,height,sample_rate,channels,duration",
            "-of", "json",
            file_path
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            # bubble error up with stderr for debugging
            raise RuntimeError(f"ffprobe failed (rc={proc.returncode}): {proc.stderr.strip()}")

        try:
            info = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise RuntimeError("ffprobe returned non-JSON output")

        streams = info.get("streams", [])
        parsed_streams = []
        for s in streams:
            parsed = {
                "index": s.get("index"),
                "codec_type": s.get("codec_type"),
                "codec_name": s.get("codec_name"),
                "codec_long_name": s.get("codec_long_name"),
                # width/height only for video
                "width": s.get("width"),
                "height": s.get("height"),
                # audio fields
                "sample_rate": s.get("sample_rate"),
                "channels": s.get("channels"),
                # duration might be available
                "duration": s.get("duration")
            }
            parsed_streams.append(parsed)

        return len(parsed_streams), parsed_streams

    @staticmethod
    def convert_video_codec(input_path: str, output_dir: str, codec: str) -> str:
        """
        Convert a video to a specific codec (VP8, VP9, H.265/HEVC, or AV1).
        Returns the path to the converted video file.
        
        Args:
            input_path: Path to the input video file
            output_dir: Directory where output files will be saved
            codec: Target codec - 'vp8', 'vp9', 'h265', or 'av1'
        
        Returns:
            Path to the converted video file
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Codec mapping: codec name -> (encoder, extension, extra_args)
        codec_map = {
            'vp8': ('libvpx', '.webm', ['-b:v', '1M', '-crf', '10']),
            'vp9': ('libvpx-vp9', '.webm', ['-b:v', '0', '-crf', '30']),
            'h265': ('libx265', '.mp4', ['-crf', '28', '-preset', 'medium']),
            'av1': ('libaom-av1', '.mkv', ['-crf', '30', '-b:v', '0', '-cpu-used', '4'])
        }
        
        codec_lower = codec.lower()
        if codec_lower not in codec_map:
            raise ValueError(f"Unsupported codec '{codec}'. Supported: {', '.join(codec_map.keys())}")
        
        encoder, extension, extra_args = codec_map[codec_lower]
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_{codec_lower}{extension}")
        
        # Build ffmpeg command
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', encoder
        ] + extra_args + [
            '-c:a', 'copy' if codec_lower in ['h265'] else 'libopus',
            output_path
        ]
        
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion to {codec} failed: {proc.stderr}")
        
        return output_path

    @staticmethod
    def create_encoding_ladder(input_path: str, output_dir: str, codec: str = 'h265') -> List[Dict[str, str]]:
        """
        Create an encoding ladder with multiple resolutions and bitrates.
        Internally calls resize() and convert_video_codec() methods to avoid code repetition.
        
        Args:
            input_path: Path to the input video file
            output_dir: Directory where output files will be saved
            codec: Target codec for encoding (default: h265)
        
        Returns:
            List of dicts with resolution, bitrate, and file path for each variant
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Define encoding ladder rungs: (width, height, bitrate_kbps, label)
        ladder_rungs = [
            (1920, 1080, 5000, '1080p'),
            (1280, 720, 2800, '720p'),
            (854, 480, 1400, '480p'),
            (640, 360, 800, '360p'),
            (426, 240, 400, '240p')
        ]
        
        # Codec settings
        codec_lower = codec.lower()
        if codec_lower == 'h265':
            encoder = 'libx265'
            extension = '.mp4'
            audio_codec = 'aac'
        elif codec_lower == 'vp9':
            encoder = 'libvpx-vp9'
            extension = '.webm'
            audio_codec = 'libopus'
        elif codec_lower == 'av1':
            encoder = 'libaom-av1'
            extension = '.mkv'
            audio_codec = 'libopus'
        else:
            # Default to h264 for broader compatibility
            encoder = 'libx264'
            extension = '.mp4'
            audio_codec = 'aac'
        
        results = []
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        for width, height, bitrate, label in ladder_rungs:
            try:
                # Step 1: Resize using existing resize method
                resized_temp = os.path.join(output_dir, f"temp_resized_{label}_{base_name}.mp4")
                FFmpegAuto.resize(input_path, width, height, resized_temp)
                
                # Step 2: Encode with target codec and bitrate
                output_path = os.path.join(output_dir, f"{base_name}_{label}_{codec_lower}{extension}")
                
                cmd = [
                    'ffmpeg', '-y', '-i', resized_temp,
                    '-c:v', encoder,
                    '-b:v', f'{bitrate}k',
                    '-maxrate', f'{int(bitrate * 1.5)}k',
                    '-bufsize', f'{int(bitrate * 2)}k',
                    '-c:a', audio_codec,
                    '-b:a', '128k',
                    output_path
                ]
                
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    # If encoding fails, log but continue with other rungs
                    print(f"Warning: Failed to encode {label}: {proc.stderr}")
                    continue
                
                # Clean up temporary resized file
                if os.path.exists(resized_temp):
                    os.remove(resized_temp)
                
                results.append({
                    'resolution': label,
                    'width': width,
                    'height': height,
                    'bitrate': f'{bitrate}kbps',
                    'codec': codec_lower,
                    'file_path': output_path,
                    'file_name': os.path.basename(output_path)
                })
                
            except Exception as e:
                print(f"Error creating {label} variant: {str(e)}")
                # Clean up temp file if it exists
                if 'resized_temp' in locals() and os.path.exists(resized_temp):
                    os.remove(resized_temp)
                continue
        
        if not results:
            raise RuntimeError("Failed to create any encoding ladder variants")
        
        return results

    

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
