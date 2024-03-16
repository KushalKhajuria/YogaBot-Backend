import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from functions import *

class Video:
    def __init__(self):
        self.model_path = 'face_landmarker_v2_with_blendshapes.task'
        self.base_options = mp.tasks.BaseOptions
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = self.FaceLandmarkerOptions(
            base_options=self.base_options(model_asset_path=self.model_path),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )

        self.landmarker = self.FaceLandmarker.create_from_options(self.options)

    def runVideo(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            
            ret, frame = cap.read()

            image = cv2.flip(frame, 1)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            
            face_landmarker_result = self.landmarker.detect(image)
            
            image = draw_landmarks_on_image(image.numpy_view(), face_landmarker_result)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.namedWindow('image', cv2.WINDOW_NORMAL) #cv2.WINDOW_AUTOSIZE or cv2.WINDOW_NORMAL if you want to resize
            cv2.resizeWindow('image', 1440, 810)

            start_point = (640, 0)
            end_point = (640, 720)
            color = (0, 255, 0)
            thickness = 9

            image = cv2.line(image, start_point, end_point, color, thickness)

            color = (0, 0, 255)
            yes_point = (240, 360)
            no_point = (900, 360)
            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(image, 'YES', yes_point, font, 3, color, 3, cv2.LINE_AA)

            cv2.putText(image, 'NO', no_point, font, 3, color, 3, cv2.LINE_AA)
            
            cv2.imshow('image', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()        

