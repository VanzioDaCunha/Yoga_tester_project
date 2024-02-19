import cv2
import time
import mediapipe as mp
import numpy as np
import tensorflow as tf
from keras.models import load_model

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

avg = 0
count = 0
label = 'null'
keypoints = np.empty((1, 133))
init_time = time.time()

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("no video in frame")
            break

        start_time = time.time()
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (500, 700))
        result = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        frame_time = init_time - start_time

        if count < 4:
            data = [frame_time]

            # Iterate over landmarks
            if result.pose_landmarks:
                for landmark in result.pose_landmarks.landmark:
                    data.append(float(landmark.x))
                    data.append(float(landmark.y))
                    data.append(float(landmark.z))
                    data.append(float(landmark.visibility))
            data = np.array(data)
            data = data.reshape(-1, 133)
            keypoints = np.concatenate((keypoints, data))

        else:
            # converts the key points data into data frames for classifier
            keypoints = np.delete(keypoints, 0, axis=0)
            keypoints = np.delete(keypoints, 0, axis=0)

            data = [frame_time]

            # Iterate over landmarks
            if result.pose_landmarks:
                for landmark in result.pose_landmarks.landmark:
                    data.append(float(landmark.x))
                    data.append(float(landmark.y))
                    data.append(float(landmark.z))
                    data.append(float(landmark.visibility))
            data = np.array(data)
            data = data.reshape(-1, 133)
            keypoints = np.concatenate((keypoints, data))
            print(keypoints.shape)
            # using the model to classify

            num_samples = keypoints.shape[0] // 4  # Calculate the number of samples after aggregation
            keypoints = keypoints[:num_samples * 4].reshape(-1, 4, 133)

            model_path = 'modelname.h5'
            classifier = load_model(model_path)
            label = classifier.predict(keypoints)



        # Displays the result to the Screen
        cv2.imshow('media pipe pose', cv2.flip(image, 1))
        print(label)

        # Calculate the Average fps of the model
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        avg += fps
        count += 1

        if cv2.waitKey(5) & 0xFF == 27:
            break


print("average fps is ", avg / count)
cap.release()
