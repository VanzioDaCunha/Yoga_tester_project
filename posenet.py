import numpy as np
import tensorflow as tf
import cv2
import math
import time
from PIL import Image, ImageOps
from constants import VIDEO_FILE_PATH, VIDEO_FILE

avg = 0
count = 0
interpreter = tf.lite.Interpreter(model_path='1.tflite')

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

parts_to_compare = [(5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (11, 12), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15),
                    (14, 16)]

i = VIDEO_FILE_PATH + VIDEO_FILE
cap = cv2.VideoCapture(1)


def parse_output(heatmap, offset, threshold):
    joint_num = heatmap.shape[-1]
    pose_kps = np.zeros((joint_num, 3), np.uint32)

    for i in range(heatmap.shape[-1]):
        joint_heatmap = heatmap[..., i]
        max_val_pos = np.squeeze(np.argwhere(joint_heatmap == np.max(joint_heatmap)))
        remap = np.array(max_val_pos /8 * 127.5, dtype=np.int32)
        pose_kps[i, 0] = (remap[0] + offset[max_val_pos[0], max_val_pos[1], i]).astype(np.uint32)
        pose_kps[i, 1] = (remap[1] + offset[max_val_pos[0], max_val_pos[1], i]).astype(np.uint32)
        max_prob = np.max(joint_heatmap)
        if max_prob > threshold:
            if pose_kps[i, 0] < 257 and pose_kps[i, 1] < 257:
                pose_kps[i, 2] = 1

    return pose_kps


def draw_kps(show_img, kps, ratio=None):
    for i in range(0, kps.shape[0]):
        if kps[i, 2]:
            if isinstance(ratio, tuple):
                cv2.circle(show_img, (int(round(kps[i, 1] * ratio[1])), int(round(kps[i, 0] * ratio[0]))), 2,
                           (0, 255, 255), round(int(1 * ratio[1])))
                continue
            cv2.circle(show_img, (kps[i, 1], kps[i, 0]), 2, (0, 255, 255), -1)

    return show_img


def draw_lines(img, keypoints, pairs):
    keypoints = keypoints.astype('int')

    for i, pair in enumerate(pairs):
        color = (0, 255, 0)
        cv2.line(img, (keypoints[pair[0]][1], keypoints[pair[0]][0]), (keypoints[pair[1]][1], keypoints[pair[1]][0]),
                color=color, lineType=cv2.LINE_AA, thickness=1)


while True:
    while cap.isOpened():
        sucess, img = cap.read()
        start_time = time.time()
        img = Image.fromarray(img)
        img_resized = ImageOps.fit(img, (width, height), Image.Resampling.LANCZOS)
        image = np.array(img_resized).reshape(height, width, 3)
        image = np.expand_dims(image.copy(), axis=0)

        floating_model = input_details[0]['dtype'] == np.float32

        if floating_model:
            image = (np.float32(image) - 127.5) / 127.5

        interpreter.set_tensor(input_details[0]['index'], image)

        interpreter.invoke()
        output = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
        output_offset = np.squeeze(interpreter.get_tensor(output_details[1]['index']))

        temp_image = np.squeeze((image.copy() * 127.5 + 127.5) / 255.0)
        temp_image = np.array(temp_image * 255, np.uint8)

        temp_kps = parse_output(output, output_offset, 0.35)
        cv2.imshow('image', draw_kps(temp_image.copy(), temp_kps))
        cv2.waitKey(10)
        draw_lines(temp_image, temp_kps, parts_to_compare)
        cv2.imshow('image1', temp_image)
        cv2.waitKey(10)

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        avg += fps
        count += 1
        print("average fps is ", avg / count)

