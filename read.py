import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import subprocess

DATA_DIR = '/Users/richard/Desktop/'

renderer_path = '../PanoRenderer/build/main'
input_vid_path = DATA_DIR + 'Dataset/v035-1.mp4'
trajs_path = 'trajs.npy'
output_vid_path = 'out.mp4'


cap = cv2.VideoCapture(input_vid_path)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('n_frames:', n_frames)

cnt = 0
for i in tqdm(range(n_frames)):
    ret, frame = cap.read()
    if not ret:
        break
    cnt += 1

print('cnt:', cnt)