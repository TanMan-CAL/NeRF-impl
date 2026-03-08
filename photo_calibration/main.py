# calibration/main.py
from calibration import calibrate_camera
from pose_estimation import estimate_camera_poses
from undistort import undistort_images
from dataset import save_dataset

calib_path = "calib_images/*.jpg"
object_path = "object_images/*.jpg"

camera_matrix, dist_coeffs = calibrate_camera(calib_path, tag_size=0.02)

c2w_matrices = estimate_camera_poses(object_path, camera_matrix, dist_coeffs, tag_size=0.02)

undistorted_images = undistort_images(object_path, camera_matrix, dist_coeffs)

n = len(undistorted_images)
n_train = int(n * 0.8)

save_dataset(
    "my_nerf_data.npz",
    images     = undistorted_images[:n_train],
    c2ws       = c2w_matrices[:n_train],
    focal      = camera_matrix[0, 0],
    images_val = undistorted_images[n_train:],
    c2ws_val   = c2w_matrices[n_train:],
)