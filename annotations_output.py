"""
Created by Vanzio on 11/02/2024
last modified: 11/02/2024 22:37
"""


import csv
import os

# header list and the format for the output file
header_list = ['Timestamp',
               'nose_x', 'nose_Y', 'nose_Z', 'nose_visibility',
               'left_eye_i_x', 'left_eye_i_Y', 'left_eye_i_Z', 'left_eye_i_visibility',
               'left_eye_x', 'left_eye_Y', 'left_eye_Z', 'left_eye_visibility',
               'left_eye_o_x', 'left_eye_o_Y', 'left_eye_o_Z', 'left_eye_o_visibility',
               'right_eye_i_x', 'right_eye_i_Y', 'right_eye_i_Z', 'right_eye_i_visibility',
               'right_eye_x', 'right_eye_Y', 'right_eye_Z', 'right_eye_visibility',
               'right_eye_o_x', 'right_eye_o_Y', 'right_eye_o_Z', 'right_eye_o_visibility',
               'left_ear_x', 'left_ear_Y', 'left_ear_Z', 'left_ear_visibility',
               'right_ear_x', 'right_ear_Y', 'right_ear_Z', 'right_ear_visibility',
               'left_mouth_x', 'left_mouth_Y', 'left_mouth_Z', 'left_mouth_visibility',
               'right_mouth_x', 'right_mouth_Y', 'right_mouth_Z', 'right_mouth_visibility',
               'left_shoulder_x', 'left_shoulder_Y', 'left_shoulder_Z', 'left_shoulder_visibility',
               'right_shoulder_x', 'right_shoulder_Y', 'right_shoulder_Z', 'right_shoulder_visibility',
               'left_elbow_x', 'left_elbow_Y', 'left_elbow_Z', 'left_elbow_visibility',
               'right_elbow_x', 'right_elbow_Y', 'right_elbow_Z', 'right_elbow_visibility',
               'left_wrist_x', 'left_wrist_Y', 'left_wrist_Z', 'left_wrist_visibility',
               'right_wrist_x', 'right_wrist_Y', 'right_wrist_Z', 'right_wrist_visibility',
               'left_pinky_x', 'left_pinky_Y', 'left_pinky_Z', 'left_pinky_visibility',
               'right_pinky_x', 'right_pinky_Y', 'right_pinky_Z', 'right_pinky_visibility',
               'left_index_x', 'left_index_Y', 'left_index_Z', 'left_index_visibility',
               'right_index_x', 'right_index_Y', 'right_index_Z', 'right_index_visibility',
               'left_thumb_x', 'left_thumb_Y', 'left_thumb_Z', 'left_thumb_visibility',
               'right_thumb_x', 'right_thumb_Y', 'right_thumb_Z', 'right_thumb_visibility',
               'left_hip_x', 'left_hip_Y', 'left_hip_Z', 'left_hip_visibility',
               'right_hip_x', 'right_hip_Y', 'right_hip_Z', 'right_hip_visibility',
               'left_knee_x', 'left_knee_Y', 'left_knee_Z', 'left_knee_visibility',
               'right_knee_x', 'right_knee_Y', 'right_knee_Z', 'right_knee_visibility',
               'left_ankle_x', 'left_ankle_Y', 'left_ankle_Z', 'left_ankle_visibility',
               'right_ankle_x', 'right_ankle_Y', 'right_ankle_Z', 'right_ankle_visibility',
               'left_heel_x', 'left_heel_Y', 'left_heel_Z', 'left_heel_visibility',
               'right_heel_x', 'right_heel_Y', 'right_heel_Z', 'right_heel_visibility',
               'left_foot_index_x', 'left_foot_index_Y', 'left_foot_index_Z', 'left_foot_index_visibility',
               'right_foot_index_x', 'right_foot_index_Y', 'right_foot_index_Z', 'right_foot_index_visibility',
               'Activity'
               ]


# Function takes input of the results class and will add all the keypoint data to the csv file
# Also take input for timestamp and the output class
def write_key(results, time=0, activity='no activity'):
    # Open CSV file for writing (replace 'output.csv' with your desired filename)
    with open('output.csv', 'a') as file:
        writer = csv.writer(file)

        data = [time]

        # Iterate over landmarks
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                data.append(landmark.x)
                data.append(landmark.y)
                data.append(landmark.z)
                data.append(landmark.visibility)
            data.append(activity)

        # Write landmark data to CSV
        writer.writerow(data)


# Function setup's the output file with the meta data
def init_file(file):
    # checks if the file already exists
    if not os.path.isfile(file):
        with open('output.csv', 'w', newline='') as file:
            writer = csv.writer(file)

            # Write header row (adjust columns based on your data)
            writer.writerow(header_list)
        print("file has been created")
    else:
        print("file already exists")


# init_file('output.csv')
