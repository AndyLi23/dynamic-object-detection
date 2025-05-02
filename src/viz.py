import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from tqdm import tqdm

OUT_WIDTH_RATIO = 2.5
OUT_HEIGHT_RATIO = 1.5

def viz_optical_flow_img(flow):
    hsv = np.zeros((*flow.shape[:-1], 3), dtype=np.uint8)
    valid_mask = ~np.isnan(flow[..., 0]) & ~np.isnan(flow[..., 1])

    magnitude, angle = cv.cartToPolar(flow[..., 0][valid_mask], flow[..., 1][valid_mask])

    hsv[..., 0][valid_mask] = (angle * 180 / np.pi / 2).flatten() # Hue corresponds to direction. OpenCV convention is 0-179
    hsv[..., 1][valid_mask] = 255  # Saturation is set to maximum
    hsv[..., 2][valid_mask] = (cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)).flatten()  # Value corresponds to magnitude
    hsv[..., 2][~valid_mask] = 255
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

def viz_optical_flow(flow):
    flow_image = viz_optical_flow_img(flow)

    plt.figure(figsize=(10, 10))
    plt.imshow(flow_image)
    plt.title("Optical Flow")
    plt.axis("off")
    plt.show()

def viz_optical_flow_diff(magnitude_diff, angle_diff):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(magnitude_diff, cmap='hot')
    plt.title("Magnitude Difference")

    plt.colorbar(label="Scale")

    plt.subplot(1, 2, 2)
    plt.imshow(angle_diff, cmap='hsv')
    plt.title("Angle Difference")

    plt.show()

def viz_optical_flow_diff_batch(N, geometric_flow_batch, raft_flow_batch, image_batch, magnitude_diff_batch, norm_magnitude_diff_batch, angle_diff_batch, fps, plt_dpi=100, output='optical_flow_diff.avi'):
    height, width = magnitude_diff_batch[0].shape
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    vid_size = (int(width * OUT_WIDTH_RATIO), int(height * OUT_HEIGHT_RATIO))
    out = cv.VideoWriter(output, fourcc, fps, vid_size)

    print(f'saving optical flow difference video to {output}...')

    figsize = (vid_size[0] / plt_dpi, vid_size[1] / plt_dpi)

    for i in tqdm(range(N)):
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        axes[0][0].imshow(image_batch[i])
        axes[0][0].set_title("Image")

        im1 = axes[0][1].imshow(magnitude_diff_batch[i], cmap='hot')
        axes[0][1].set_title("Magnitude Difference")
        cbar1 = fig.colorbar(im1, ax=axes[0][1], orientation='vertical', fraction=0.046, pad=0.04)
        cbar1.set_label("Scale")

        im2 = axes[0][2].imshow(norm_magnitude_diff_batch[i], cmap='hot')
        axes[0][2].set_title("Normalized Magnitude Difference")
        cbar2 = fig.colorbar(im2, ax=axes[0][2], orientation='vertical', fraction=0.046, pad=0.04)
        cbar2.set_label("Scale")

        im3 = axes[1][0].imshow(angle_diff_batch[i], cmap='hsv')
        axes[1][0].set_title("Angle Difference")
        cbar3 = fig.colorbar(im3, ax=axes[1][0], orientation='vertical', fraction=0.046, pad=0.04)
        cbar3.set_label("Angle")

        axes[1][1].imshow(viz_optical_flow_img(geometric_flow_batch[i]))
        axes[1][1].set_title("Geometric Optical Flow")

        axes[1][2].imshow(viz_optical_flow_img(raft_flow_batch[i]))
        axes[1][2].set_title("RAFT Optical Flow")

        fig.tight_layout() 
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frame = frame[:, :, [1, 2, 3]]  # Convert ARGB to RGB by dropping the alpha channel
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        frame = cv.resize(frame, vid_size)
        out.write(frame)

        plt.close(fig)

    print('video saved successfully')

    out.release()