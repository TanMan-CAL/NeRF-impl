# calibration/undistort.py
import cv2
import numpy as np
import glob

def undistort_images(object_images_path, camera_matrix, dist_coeffs):
    undistorted_images = []

    for fname in glob.glob(object_images_path):
        img = cv2.imread(fname)
        h, w = img.shape[:2]

        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), alpha=0)
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

        x, y, w_roi, h_roi = roi
        undistorted = undistorted[y:y+h_roi, x:x+w_roi]

        undistorted_images.append(undistorted)

    return undistorted_images
