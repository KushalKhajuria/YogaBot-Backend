#@markdown We implemented some functions to visualize the face landmark detection results. <br/> Run the following cell to activate the functions.

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt


def draw_landmarks_on_image_face(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for face_landmarks in face_landmarks_list:
    # Create a NormalizedLandmarkList
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    # Draw face landmarks with different styles
    for connections, style in [
        (mp.solutions.face_mesh.FACEMESH_TESSELATION, mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()),
        (mp.solutions.face_mesh.FACEMESH_CONTOURS, mp.solutions.drawing_styles.get_default_face_mesh_contours_style()),
        (mp.solutions.face_mesh.FACEMESH_IRISES, mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())
    ]:
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=connections,
            # landmark_drawing_spec=None,
            connection_drawing_spec=style
        )

  # Return the annotated image
  return annotated_image

def draw_landmarks_on_image_body(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for pose_landmarks in pose_landmarks_list:
    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])

    # for connections, style in [
    #   (mp.solutions.pose.POSE_CONNECTIONS, mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    # ]:
    #   for key, value in style.items():
    #       print(key, ";", value)
    #
    #   solutions.drawing_utils.draw_landmarks(
    #       image=annotated_image,
    #       landmark_list=pose_landmarks_proto,
    #       connections=connections,
    #       connection_drawing_spec=style
    #   )

    # for key, value in mp.solutions.drawing_styles.get_default_pose_landmarks_style().items():
    #    print(key, ";", value)

    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())

  # Return the annotated image
  return annotated_image 
