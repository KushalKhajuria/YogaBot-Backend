# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from functions import *
import cv2

# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("image2.jpg") #i changed it from image.png

# STEP 4: Detect face landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. In this case, visualize it.

annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

print(f"image: {image}, result: {type(detection_result)}, annotated_image: {type(annotated_image)}")

cv2.imshow('image',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

cv2.waitKey(0)

cv2.destroyAllWindows()

#plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])

#print(detection_result.facial_transformation_matrixes)

