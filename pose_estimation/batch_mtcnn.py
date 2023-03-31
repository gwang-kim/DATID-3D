import argparse
import cv2
import os
from mtcnn import MTCNN
import random
from tqdm import tqdm
import numpy as np

detector = MTCNN()
parser = argparse.ArgumentParser()
parser.add_argument('--in_root', type=str, default="", help='process folder')
args = parser.parse_args()
in_root = args.in_root

out_root = os.path.join(in_root, "debug")
out_detection = os.path.join(in_root, "detections")
if not os.path.exists(out_root):
    os.makedirs(out_root)
if not os.path.exists(out_detection):
    os.makedirs(out_detection)

imgs = sorted([x for x in os.listdir(in_root) if x.endswith(".jpg") or x.endswith(".png")])
random.shuffle(imgs)
for img in tqdm(imgs):
    src = os.path.join(in_root, img)
    dst = os.path.join(out_detection, img.replace(".jpg", ".txt").replace(".png", ".txt"))

    if not os.path.exists(dst):
        image = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
        print(image.shape)
        result = detector.detect_faces(image)

        if len(result)>0:
            index = 0
            if len(result)>1: # if multiple faces, take the biggest face
                # size = -100000
                lowest_dist = float('Inf')
                for r in range(len(result)):
                    # print(result[r]["box"][0], result[r]["box"][1])
                    face_pos = np.array(result[r]["box"][:2]) + np.array(result[r]["box"][2:])/2

                    dist_from_center = np.linalg.norm(face_pos - np.array([1500./2, 1500./2]))
                    if dist_from_center < lowest_dist:
                        lowest_dist = dist_from_center
                        index=r


                    # size_ = result[r]["box"][2] + result[r]["box"][3]
                    # if size < size_:
                    #     size = size_
                    #     index = r

            # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
            bounding_box = result[index]['box']
            keypoints = result[index]['keypoints']
            if result[index]["confidence"] > 0.9:

                cv2.rectangle(image,
                            (bounding_box[0], bounding_box[1]),
                            (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                            (0,155,255),
                            2)

                cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
                cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
                cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
                cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
                cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

                dst = os.path.join(out_root, img)
                # cv2.imwrite(dst, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                dst = os.path.join(out_detection, img.replace(".jpg", ".txt").replace(".png", ".txt"))
                outLand = open(dst, "w")
                outLand.write(str(float(keypoints['left_eye'][0])) + " " + str(float(keypoints['left_eye'][1])) + "\n")
                outLand.write(str(float(keypoints['right_eye'][0])) + " " + str(float(keypoints['right_eye'][1])) + "\n")
                outLand.write(str(float(keypoints['nose'][0])) + " " +      str(float(keypoints['nose'][1])) + "\n")
                outLand.write(str(float(keypoints['mouth_left'][0])) + " " + str(float(keypoints['mouth_left'][1])) + "\n")
                outLand.write(str(float(keypoints['mouth_right'][0])) + " " + str(float(keypoints['mouth_right'][1])) + "\n")
                outLand.close()