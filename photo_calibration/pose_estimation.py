# calibration/pose_estimation.py
import cv2
import numpy as np
import glob

def estimate_camera_poses(object_images_path, camera_matrix, dist_coeffs, tag_size=0.02):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()

    object_points_tag = np.array([
        [0, 0, 0],
        [tag_size, 0, 0],
        [tag_size, tag_size, 0],
        [0, tag_size, 0]
    ], dtype=np.float32)

    c2w_matrices = []

    for fname in glob.glob(object_images_path):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        if ids is None:
            continue

        img_points = corners[0].reshape(-1, 2)
        success, rvec, tvec = cv2.solvePnP(object_points_tag, img_points, camera_matrix, dist_coeffs)

        if not success:
            continue

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)
        w2c = np.vstack([np.hstack([R, t]), [0,0,0,1]])
        c2w = np.linalg.inv(w2c)
        c2w_matrices.append(c2w)

    # print(f"Estimated poses for {len(c2w_matrices)} images.")
    return c2w_matrices
