import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def viz_optical_flow(flow):
    hsv = np.zeros((*flow.shape[:-1], 3), dtype=np.uint8)
    valid_mask = ~np.isnan(flow[..., 0]) & ~np.isnan(flow[..., 1])

    magnitude, angle = cv.cartToPolar(flow[..., 0][valid_mask], flow[..., 1][valid_mask])

    hsv[..., 0][valid_mask] = (angle * 180 / np.pi / 2).flatten() # Hue corresponds to direction. OpenCV convention is 0-179
    hsv[..., 1][valid_mask] = 255  # Saturation is set to maximum
    hsv[..., 2][valid_mask] = (cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)).flatten()  # Value corresponds to magnitude
    hsv[..., 2][~valid_mask] = 255
    flow_image = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    plt.figure(figsize=(10, 10))
    plt.imshow(flow_image)
    plt.title("Optical Flow")
    plt.axis("off")
    plt.show()