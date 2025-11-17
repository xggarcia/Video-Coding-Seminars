import ffmpeg
from os import remove
import os

try:
    remove("output.jpg")
except FileNotFoundError:
    pass

# (

#     ffmpeg
#     .input('image_to_crop.jpg')
#     .filter('threshold', '128', '255', 'black', 'white')
#     .output('output.jpg')
#     .run()
# )




file_path = "input.jpg"
output_path = "output.jpg"


os.system('ffmpeg -i {file_path} -f lavfi -i \
        color=gray:s=1280x720 -f lavfi -i \
        color=black:s=1280x720 -f lavfi -i \
        color=white:s=1280x720 -filter_complex \
        threshold {output_path}')
