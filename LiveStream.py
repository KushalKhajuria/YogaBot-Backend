import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from functions import *
import time

class FaceLandmarkerHandler:
    def __init__(self):
        self.model_path = 'face_landmarker_v2_with_blendshapes.task'
        self.base_options = mp.tasks.BaseOptions
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = self.FaceLandmarkerOptions(
            base_options=self.base_options(model_asset_path=self.model_path),
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            #min_face_detection_confidence = 0.5,
            #min_face_presence_confidence = 0.5,
            #min_tracking_confidence = 0.5,
            result_callback = self.print_result
        )

        self.detector = self.FaceLandmarker.create_from_options(self.options)

    def detect_faces_async(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()

            timestamp_ms = int(time.time() * 1000)
            print(f"timestamp_ms: {timestamp_ms}")

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

            self.detector.detect_async(image, timestamp_ms)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def print_result(self, result: mp.tasks.vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):

        #result = self.detector.detect(output_image)

        try:
            image = draw_landmarks_on_image(output_image.numpy_view(), result)
        except:
            print("Error with draw_landmarks_on_image")

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        print(f"In print_result :), output image: {output_image}, result: {type(result)}, image: {type(image)}")
        
        cv2.imshow('image', image)

        #cv2.waitKey(0)

# Add other necessary imports and functions from functions.py here

