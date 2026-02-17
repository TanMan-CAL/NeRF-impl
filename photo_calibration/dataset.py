# calibration/dataset.py
import numpy as np

def save_dataset(filename, images, c2ws, focal):
    images_array = np.array(images, dtype=np.uint8)
    c2ws_array = np.array(c2ws, dtype=np.float32)

    np.savez(
        filename,
        images_train=images_array,
        c2ws_train=c2ws_array,
        focal=focal
    )
