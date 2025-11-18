import ffmpeg
from os import remove
import os
import cv2

try:
    remove("output.jpg")
except FileNotFoundError:
    pass


file_path = "input.jpg"
output_path = "output.jpg"



array = ["A","A","A","A","A","B","B","B","C","C","D","F","C","C","C","D","D","C", "C"]


output = []
counter = 1
for i in range(len(array)):
    if(i == len(array)-1):
        output.append(array[i])
        output.append(counter)

    elif(array[i] == array[i+1]):
        counter += 1
        
    else:
        output.append(array[i])
        output.append(counter)
        counter = 1
        
        
        
print(output)