import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def viz_optical_flow(flow):
    hsv = np.zeros((*flow.shape[:-1], 3), dtype=np.uint8)
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

    hsv[..., 0] = angle * 180 / np.pi / 2 # Hue corresponds to direction. OpenCV convention is 0-179
    hsv[..., 1] = 255  # Saturation is set to maximum
    hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)  # Value corresponds to magnitude
    flow_image = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    plt.figure(figsize=(10, 10))
    plt.imshow(flow_image)
    plt.title("Optical Flow")
    plt.axis("off")
    plt.show()