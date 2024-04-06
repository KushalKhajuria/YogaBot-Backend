import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from functions import *
import json
import requests

class Video:
    def __init__(self):
        self.base_options = mp.tasks.BaseOptions
        
        self.face_model_path = 'face_landmarker_v2_with_blendshapes.task'
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

        self.face_options = self.FaceLandmarkerOptions(
            base_options=self.base_options(model_asset_path=self.face_model_path),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )

        self.face_landmarker = self.FaceLandmarker.create_from_options(self.face_options)

        #***********************

        self.pose_model_path = 'pose_landmarker.task'
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        self.PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

        self.pose_options = self.PoseLandmarkerOptions(
            base_options=self.base_options(model_asset_path=self.pose_model_path),
            output_segmentation_masks = True
        )
        
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(self.pose_options)

    def runVideo(self):
        cap = cv2.VideoCapture(0)

        #with mp_holistic.Holistic(
        #    min_detection_confidence=0.5,
        #    min_tracking_confidence=0.5) as holistic:

        #or tune parameters otherwise

        while cap.isOpened():
            ret, frame = cap.read()
            image = cv2.flip(frame, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            
            face_landmarker_result = self.face_landmarker.detect(image)
            pose_landmarker_result = self.pose_landmarker.detect(image)

            image1 = draw_landmarks_on_image_face(image.numpy_view(), face_landmarker_result)
            image2 = draw_landmarks_on_image_body(image.numpy_view(), pose_landmarker_result)

            image = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.namedWindow('image', cv2.WINDOW_NORMAL) #cv2.WINDOW_AUTOSIZE or cv2.WINDOW_NORMAL if you want to resize
            cv2.resizeWindow('image', 1440, 810)

            #start_point = (640, 0)
            #end_point = (640, 720)
            #color = (0, 255, 0)
            #thickness = 9

            #image = cv2.line(image, start_point, end_point, color, thickness)

            #color = (0, 0, 255)
            #yes_point = (240, 360)
            #no_point = (900, 360)
            #font = cv2.FONT_HERSHEY_SIMPLEX

            #cv2.putText(image, 'YES', yes_point, font, 3, color, 3, cv2.LINE_AA)

            #cv2.putText(image, 'NO', no_point, font, 3, color, 3, cv2.LINE_AA)
            
            cv2.imshow('image', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                #edit this to download image for experimental part of project

                cv2.imwrite('image.jpg', image)

                #print(face_landmarker_result.face_blendshapes)
                #print("\n\n\n")
                #print(face_landmarker_result.face_landmarks)
                #print("\n\n\n")
                print("landmarks" + str(pose_landmarker_result.pose_landmarks))
                #print(face_landmarker_result.facial_transformation_matrixes)
                #print(face_landmarker_result.timestampMs)

                spring_boot_url = "http://localhost:8080/test/numeric_data"

                landmarks_xy = []

                for l in pose_landmarker_result.pose_landmarks:
                    for landmark in l:
                        landmarks_xy.append([landmark.x, landmark.y])

                data_payload = json.dumps(landmarks_xy)

                headers={
                    'Content-type':'application/json',
                    'Accept':'application/json'
                }

                response = requests.post(
                    spring_boot_url,
                    data=data_payload,
                    headers = headers)

                print("data_payload" + data_payload)
                print("landmarks_xy" + str(landmarks_xy))
                print("response" + str(response))

                break

        cap.release()
        cv2.destroyAllWindows()        

