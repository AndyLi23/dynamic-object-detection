import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
from tqdm import tqdm
from itertools import product
import gc

PLT_DPI = 100


class OpticalFlowVisualizer:
    def __init__(self, viz_params, output, fps):
        self.params = viz_params
        if not self.params.viz_video: return

        # matplotlib.use('Agg')

        fourcc = cv.VideoWriter_fourcc(*'XVID')
        self.plt_shape = self.params.viz_flags.shape
        self.figsize = (self.params.vid_dims[0] / PLT_DPI, self.params.vid_dims[1] / PLT_DPI)
        self.output_file = output
        self.video_writer = cv.VideoWriter(output, fourcc, fps, self.params.vid_dims)

    def write_batch_frames(self, image, depth, dynamic_mask, orig_dynamic_masks, raft_flow, residual):
        if not self.params.viz_video: return

        for frame in range(len(image)):
            fig, axes = plt.subplots(self.plt_shape[0], self.plt_shape[1], figsize=self.figsize)

            for i, j in product(range(self.plt_shape[0]), range(self.plt_shape[1])):
                name = self.params.viz_flag_names[self.params.viz_flags[i][j]]

                if name == 'image':
                    axes[i][j].imshow(image[frame])
                    axes[i][j].set_title("Image")
                elif name == 'depth':
                    axes[i][j].imshow(depth[frame], cmap='gray')
                    axes[i][j].set_title("Depth")
                elif name == 'dynamic mask':
                    axes[i][j].imshow(dynamic_mask[frame], cmap='gray')
                    axes[i][j].set_title("Dynamic Mask")
                elif name == 'orig_dynamic_mask':
                    axes[i][j].imshow(orig_dynamic_masks[frame], cmap='gray')
                    axes[i][j].set_title("Original Dynamic Mask")
                elif name == 'raft flow':
                    axes[i][j].imshow(OpticalFlowVisualizer.viz_optical_flow_img(raft_flow[frame]))
                    axes[i][j].set_title("RAFT Optical Flow")
                elif name == 'residual':
                    im = axes[i][j].imshow(residual[frame], cmap='hot', vmin=0, vmax=self.params.viz_max_residual_magnitude)
                    axes[i][j].set_title("Magnitude Residual")
                    cbar = fig.colorbar(im, ax=axes[i][j], orientation='vertical', fraction=0.046, pad=0.04)
                    cbar.set_label("Scale")

            fig.tight_layout()
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            buf = buf[:, :, [1, 2, 3]]  # Convert ARGB to RGB by dropping the alpha channel
            vid_frame = cv.cvtColor(buf, cv.COLOR_RGB2BGR)
            vid_frame = cv.resize(vid_frame, self.params.vid_dims)
            self.video_writer.write(vid_frame)

            plt.close(fig)
            del fig, axes, buf, vid_frame
            gc.collect()

    def end(self):
        if not self.params.viz_video: return
        print(f'video saved to {self.output_file}')
        self.video_writer.release()


    @classmethod
    def viz_optical_flow_img(cls, flow):
        hsv = np.zeros((*flow.shape[:-1], 3), dtype=np.uint8)
        valid_mask = ~np.isnan(flow[..., 0]) & ~np.isnan(flow[..., 1])

        magnitude, angle = cv.cartToPolar(flow[..., 0][valid_mask], flow[..., 1][valid_mask])

        hsv[..., 0][valid_mask] = (angle * 180 / np.pi / 2).flatten() # Hue corresponds to direction. OpenCV convention is 0-179
        hsv[..., 1][valid_mask] = 255  # Saturation is set to maximum
        hsv[..., 2][valid_mask] = (cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)).flatten()  # Value corresponds to magnitude
        hsv[..., 2][~valid_mask] = 255
        return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
   


@DeprecationWarning
def viz_optical_flow(flow):
    flow_image = OpticalFlowVisualizer.viz_optical_flow_img(flow)

    plt.figure(figsize=(10, 10))
    plt.imshow(flow_image)
    plt.title("Optical Flow")
    plt.axis("off")
    plt.show()

@DeprecationWarning
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

@DeprecationWarning
def viz_optical_flow_diff_batch(N, geometric_flow_batch, raft_flow_batch, image_batch, magnitude_diff_batch, norm_magnitude_diff_batch, angle_diff_batch, fps, 
                                output='optical_flow_diff.avi', OUT_WIDTH_RATIO=2.5, OUT_HEIGHT_RATIO=1.5):
    height, width = magnitude_diff_batch[0].shape
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    vid_size = (int(width * OUT_WIDTH_RATIO), int(height * OUT_HEIGHT_RATIO))
    out = cv.VideoWriter(output, fourcc, fps, vid_size)

    print(f'saving optical flow difference video to {output}...')

    figsize = (vid_size[0] / PLT_DPI, vid_size[1] / PLT_DPI)

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

        axes[1][1].imshow(OpticalFlowVisualizer.viz_optical_flow_img(geometric_flow_batch[i]))
        axes[1][1].set_title("Geometric Optical Flow")

        axes[1][2].imshow(OpticalFlowVisualizer.viz_optical_flow_img(raft_flow_batch[i]))
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