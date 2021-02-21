from __future__ import absolute_import
import numpy as np

from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional

import os

import PIL.Image


def align_image_and_landmarks(image, mask, dense_landmarks, landmarks, size):
    output_size=size
    transform_size=size
    lm = np.array(landmarks)
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    img = image
    mask = mask.convert("RGB")
    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        mask = mask.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        mask = mask.crop(crop)
        quad -= crop[0:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)

    # Transform.
    mask = mask.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)

    import cv2
    src = np.array(((quad[0, 0], quad[0, 1]), (quad[1, 0], quad[1, 1]),\
                    (quad[2, 0], quad[2, 1]), (quad[3, 0], quad[3, 1])), dtype=np.float32)
    dest = np.array(((0, 0), (0, transform_size-1), (transform_size-1, transform_size-1), (transform_size-1, 0)), dtype=np.float32)
    mtx = cv2.getPerspectiveTransform(src, dest)
    ones = np.ones((dense_landmarks.shape[0], 1))
    dense_landmarks_homo = np.concatenate((dense_landmarks, ones), axis=1)
    dense_landmarks_transformed = np.matmul(mtx, dense_landmarks_homo.T)
    dense_landmarks_transformed = dense_landmarks_transformed.T[:, :2]
    mask = mask.convert("L")
    
    return img, mask, dense_landmarks_transformed

