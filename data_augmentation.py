import cv2
import time
import mediapipe as mp
from annotations_input import read_csv_file, get_time
from annotations_output import write_key
from constants import CSV_FILE, CSV_FILE_PATH
from constants import VIDEO_FILE, VIDEO_FILE_PATH
import albumentations as alb
import random


def transform_keypoints(results, width, height):
    keypoints = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            x = max(0, min(landmark.x, 0.999999))
            y = max(0, min(landmark.y, 0.999999))
            x = int(float(x) * width)
            y = int(float(y) * height)
            data = [x, y]
            keypoints.append(tuple(data))

    return keypoints


def augment_keypoints(frame, keypoints):
    random.seed(22)
    transform = alb.Compose(
        [
            alb.Rotate(limit=1.5, p=1),
            alb.HorizontalFlip(p=0.5),
         ],
        keypoint_params=alb.KeypointParams(format='xy')
    )

    transformed = transform(image=frame, keypoints=keypoints)
    return transformed['image'], transformed['keypoints']


def format_keypoints(keypoint, results):
    if results.pose_landmarks and len(keypoint) == 33:
        i = 0
        for landmark in results.pose_landmarks.landmark:
            landmark.x = float(keypoint[i][0]) / 500
            landmark.y = float(keypoint[i][1]) / 700
            i += 1
    return results


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
avg: float = 0
count: int = 0

cap = cv2.VideoCapture(VIDEO_FILE_PATH + VIDEO_FILE)
annotations = read_csv_file(CSV_FILE_PATH + CSV_FILE)

with mp_pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.7) as pose:
    while cap.isOpened():

        success, image = cap.read()
        if not success:
            print("no video in frame")
            break

        frame_time = cap.get(0) / 1000
        activity = get_time(annotations, frame_time)

        start_time = time.time()
        image.flags.writeable = False

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(image)
        image = cv2.resize(image, (500, 700))

        pl = transform_keypoints(result, 500, 700)
        image, keys = augment_keypoints(image, pl)
        keys = format_keypoints(keys, result)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image,
            keys.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        write_key(result, frame_time, activity)

        cv2.imshow('media pipe pose', cv2.flip(image, 1))
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        avg += fps
        count += 1

        if cv2.waitKey(5) & 0xFF == 27:
            break

print("average fps is ", avg / count)
cap.release()
