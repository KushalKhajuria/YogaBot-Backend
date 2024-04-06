import mediapipe as mp
import cv2
from video import Video
import time

from functions import *
from openTask import *

#import asyncio

def main():
    #face_landmarker_handler = FaceLandmarkerHandler()
    #asyncio.run(face_landmarker_handler.detect_faces_async())
    #face_landmarker_handler.detect_faces_async()

    #openTask1()

    vid = Video()
    vid.runVideo()
    

if __name__ == "__main__":
    main()
