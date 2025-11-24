import subprocess
import os

def convert_gray_quantized(input_path, output_path, num_colors=4):
    """
    Converts an image to grayscale and limits it to a specific number of colors.
    
    Args:
        input_path (str): Path to source image.
        output_path (str): Path to save image (must be .png or .gif).
        num_colors (int): The maximum number of colors allowed (default 4).
    """
    
    # 1. validate extension to prevent compression artifacts
    if not output_path.lower().endswith(('.png', '.gif', '.jpg')):
        print("Warning: Output should be .png or .gif to preserve exact colors.")

    # 2. Construct the complex filter
    # We use f-strings to insert the num_colors variable
    filter_cmd = (
        f"[0:v]format=gray,split[s0][s1];"
        f"[s0]palettegen=max_colors={num_colors}[p];"
        f"[s1][p]paletteuse"
    )

    # 3. Build the full command
    command = [
        'ffmpeg',
        '-y',                 # Overwrite output without asking
        '-i', input_path,     # Input file
        '-filter_complex', filter_cmd,
        output_path
    ]

    try:
        # 4. Run the command suppressing the verbose banner
        subprocess.run(command, check=True, stderr=subprocess.DEVNULL)
        print(f"Success! Saved to {output_path} with {num_colors} colors.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

# --- Usage Example ---
if __name__ == "__main__":
    convert_gray_quantized("Seminar 1/input.jpg", "Seminar 1/output.png", num_colors=2)