import cv2
import time
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

avg = 0
count = 0
file = 'Dataset/Yoga/Trikonasana/Videos/1.mp4'
cap = cv2.VideoCapture(file)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("no video in frame")
            continue

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

        cv2.imshow('media pipe pose', cv2.flip(image, 1))
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        avg += fps
        count += 1
        #print(fps)

        if cv2.waitKey(5) & 0xFF == 27:
            break

print("average fps is ", avg / count)
cap.release()
