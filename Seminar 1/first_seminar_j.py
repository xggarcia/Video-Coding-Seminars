import os
import numpy as np
import cv2
from PIL import Image
import ffmpeg
from os import path, remove



class seminar_1:

    @staticmethod
    def RGB_YUV_translator(value_1, value_2, value_3, mode='RGB_to_YUV'):
        if mode == 'YUV_to_RGB':
            Y, U, V = value_1, value_2, value_3
            R = round(Y + 1.13983 * V, 0)
            G = round(Y - 0.39465 * U - 0.58060 * V, 0)
            B = round(Y + 2.03211 * U, 0)
            return (R, G, B)
        else:  # Default mode is 'RGB_to_YUV'
            R, G, B = value_1, value_2, value_3
            Y = round(0.299 * R + 0.587 * G + 0.114 * B, 0)
            U = round(-0.14713 * R - 0.28886 * G + 0.436 * B, 0)
            V = round(0.615 * R - 0.51499 * G - 0.10001 * B, 0)
            return (Y, U, V)
    
    @staticmethod
    def serpentine(file_path):
        # Open image and ensure RGB
        img = Image.open(file_path).convert("RGB")
        width, height = img.size
        pixels = img.load()

        serpentine_pixels = []

        for d in range(width + height - 1):

            # even diagonals go up-right
            if d % 2 == 0:
                y = min(d, height - 1)
                x = d - y
                while y >= 0 and x < width:
                    serpentine_pixels.append(pixels[x, y])
                    x += 1
                    y -= 1

            # odd diagonals go down-left
            else:
                x = min(d, width - 1)
                y = d - x
                while x >= 0 and y < height:
                    serpentine_pixels.append(pixels[x, y])
                    x -= 1
                    y += 1

        return serpentine_pixels
        
    @staticmethod
    def black_and_white(file_path, output_path):
        try:
            remove(output_path)
        except FileNotFoundError:
            pass
        
        
        os.system('ffmpeg -i ',+file_path,+' -f lavfi -i \
                color=gray:s=1280x720 -f lavfi -i \
                color=black:s=1280x720 -f lavfi -i \
                color=white:s=1280x720 -filter_complex \
                threshold',+ output_path)


        
    def resize_image(path, new_width, new_height, output_path):
        try:
            remove(output_path)
        except FileNotFoundError:
            pass
        (
            ffmpeg
            .input(path)
            .filter('scale', new_width, new_height)
            .output(output_path)
            .run()
        )
        
        
        
        
        

def main():
    print("RGB <-> YUV Translator")
    print("----------------------")

    try:
        rgb_str = input("Enter R G B values (0-255) separated by spaces (e.g. '128 64 32'): ")
        r_str, g_str, b_str = rgb_str.strip().split()
        r, g, b = int(r_str), int(g_str), int(b_str)

        translator = seminar_1()

        # RGB -> YUV
        y, u, v = translator.RGB_YUV_translator(r, g, b, mode='RGB_to_YUV')
        print(f"\nRGB ({r}, {g}, {b}) ->")
        print(f"YUV (Y={y:.2f}, U={u:.2f}, V={v:.2f})")

        # YUV -> RGB (inverse)
        r2, g2, b2 = translator.RGB_YUV_translator(y, u, v, mode='YUV_to_RGB')
        print(f"\nBack to RGB from YUV ->")
        print(f"RGB ({r2}, {g2}, {b2})")

    except ValueError:
        print("Error: please enter exactly three integer values for R, G and B.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()