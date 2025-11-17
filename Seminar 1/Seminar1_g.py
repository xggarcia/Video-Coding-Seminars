import ffmpeg
from os import remove

remove("output.jpg")
(

    ffmpeg
    .input('image_to_crop.jpg')
    .filter('scale', 10, 10)
    .output('output.jpg')
    .run()
)
