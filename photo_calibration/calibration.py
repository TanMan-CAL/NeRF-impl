# calibration/calibration.py
import cv2
import numpy as np
import glob

def calibrate_camera(calib_images_path, tag_size=0.02):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()

    object_points_1tag = np.array([
        [0, 0, 0],
        [tag_size, 0, 0],
        [tag_size, tag_size, 0],
        [0, tag_size, 0]
    ], dtype=np.float32)

    all_objpoints = []
    all_imgpoints = []
    image_size = None

    for fname in glob.glob(calib_images_path):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        if ids is not None:
            for c in corners:
                all_imgpoints.append(c.reshape(-1, 2))
                all_objpoints.append(object_points_1tag)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_objpoints, all_imgpoints, image_size, None, None
    )

    return camera_matrix, dist_coeffs