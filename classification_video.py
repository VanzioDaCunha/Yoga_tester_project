import cv2
import time
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from constants import MODEL_INPUT, LABELS, SEQUENCE_LENGTH


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

avg = 0
count = 1
index = 0
label = 'null'
keypoints = np.empty((1, MODEL_INPUT))
init_time = time.time()


model_path = 'modelname.keras'
classifier = load_model(model_path)
classifier.summary()

label_encoder = LabelEncoder()
label_encoder.fit(LABELS)

cap = cv2.VideoCapture('11.mp4')
time.sleep(1)

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

        if count % SEQUENCE_LENGTH != 0:
            time.sleep(0.1)
            # data = [frame_time]
            data = []

            # Iterate over landmarks
            if result.pose_landmarks:
                for landmark in result.pose_landmarks.landmark:
                    data.append(float(landmark.x))
                    data.append(float(landmark.y))
                    # data.append(float(landmark.z))
                    # data.append(float(landmark.visibility))
                    if landmark.visibility > 0.5:
                        data.append(1)
                    else:
                        data.append(0)
            data = np.array(data)
            data = data.reshape(-1, MODEL_INPUT)
            keypoints = np.concatenate((keypoints, data))

        else:
            keypoints = np.delete(keypoints, 0, axis=0)
            # data = [frame_time]
            data = []

            # Iterate over landmarks
            if result.pose_landmarks:
                for landmark in result.pose_landmarks.landmark:
                    data.append(float(landmark.x))
                    data.append(float(landmark.y))
                    # data.append(float(landmark.z))
                    # data.append(float(landmark.visibility))
                    if landmark.visibility > 0.5:
                        data.append(1)
                    else:
                        data.append(0)
            data = np.array(data)
            data = data.reshape(-1, MODEL_INPUT)

            keypoints = np.concatenate((keypoints, data))
            # using the model to classify

            num_samples = keypoints.shape[0] // SEQUENCE_LENGTH  # Calculate the number of samples after aggregation
            keys = keypoints[:num_samples * SEQUENCE_LENGTH].reshape(-1, SEQUENCE_LENGTH, MODEL_INPUT)

            label = classifier.predict(keys)
            # to get the class label
            cat = np.array(label[0][0])

            index = np.argmax(cat)

            # cat = label_encoder.inverse_transform(cat)
            keypoints = np.empty((1, MODEL_INPUT))

        text_size = 0.5
        text_thickness = 2
        text_position = (10, 30)

        # Displays the result to the Screen
        image = cv2.flip(image, 1)

        # Draw black outline for the text
        outline_thickness = 2
        cv2.putText(image, LABELS[index], (text_position[0] - outline_thickness, text_position[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0, 0, 0), text_thickness + 2, cv2.LINE_AA)
        cv2.putText(image, LABELS[index], (text_position[0] + outline_thickness, text_position[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0, 0, 0), text_thickness + 2, cv2.LINE_AA)
        cv2.putText(image, LABELS[index], (text_position[0], text_position[1] - outline_thickness),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0, 0, 0), text_thickness + 2, cv2.LINE_AA)
        cv2.putText(image, LABELS[index], (text_position[0], text_position[1] + outline_thickness),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0, 0, 0), text_thickness + 2, cv2.LINE_AA)

        # Draw the original text
        cv2.putText(image, LABELS[index], text_position, cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0, 255, 0), text_thickness, cv2.LINE_AA)

        cv2.imshow('media pipe pose', image)

        # Calculate the Average fps of the model
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        avg += fps
        count += 1

        if cv2.waitKey(5) & 0xFF == 27:
            break


print("average fps is ", avg / count)
cap.release()
