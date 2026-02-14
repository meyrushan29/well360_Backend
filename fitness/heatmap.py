import cv2
import numpy as np
from collections import deque


class HeatmapVisualizer:
    def __init__(self, decay=0.92, kernel_size=21, sigma=4):
        self.heatmap = None
        self.prev_landmarks = None
        self.decay = decay
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.show_heatmap = True

    def create_gaussian_kernel(self, size, sigma):
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        return kernel / kernel.sum()

    def update_heatmap(self, landmarks, frame_shape):
        h, w = frame_shape[:2]

        # Check dimensions mismatch (e.g. video resolution change)
        if self.heatmap is not None:
            if self.heatmap.shape[0] != h or self.heatmap.shape[1] != w:
                 self.heatmap = None
                 self.prev_landmarks = None

        # Apply temporal decay
        if self.heatmap is not None:
            self.heatmap *= self.decay
        else:
            self.heatmap = np.zeros((h, w), dtype=np.float32)

        if landmarks is not None:
            total_movement = 0
            if self.prev_landmarks is not None and len(landmarks) == len(self.prev_landmarks):
                for i in range(len(landmarks)):
                    dx = landmarks[i].x - self.prev_landmarks[i].x
                    dy = landmarks[i].y - self.prev_landmarks[i].y
                    total_movement += np.sqrt(dx * dx + dy * dy)

            movement_factor = min(total_movement * 120, 6.0) + 0.3

            kernel = self.create_gaussian_kernel(self.kernel_size, self.sigma)
            half_kernel = self.kernel_size // 2

            for i in range(len(landmarks)):
                x_px = int(landmarks[i].x * w)
                y_px = int(landmarks[i].y * h)

                if x_px < 0 or x_px >= w or y_px < 0 or y_px >= h:
                    continue

                x_start = max(x_px - half_kernel, 0)
                x_end = min(x_px + half_kernel + 1, w)
                y_start = max(y_px - half_kernel, 0)
                y_end = min(y_px + half_kernel + 1, h)

                kx_start = max(half_kernel - x_px, 0)
                kx_end = self.kernel_size - max(x_px + half_kernel + 1 - w, 0)
                ky_start = max(half_kernel - y_px, 0)
                ky_end = self.kernel_size - max(y_px + half_kernel + 1 - h, 0)

                if i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                    joint_weight = 1.1
                elif i in list(range(11)):
                    joint_weight = 0.2
                else:
                    joint_weight = 0.7

                self.heatmap[y_start:y_end, x_start:x_end] += (
                        kernel[ky_start:ky_end, kx_start:kx_end]
                        * movement_factor * joint_weight
                )

            self.prev_landmarks = landmarks

        return self.heatmap

    def apply_heatmap_overlay(self, frame):
        if not self.show_heatmap or self.heatmap is None:
            return frame.copy()

        # Get frame dimensions
        frame_h, frame_w = frame.shape[:2]
        heatmap_h, heatmap_w = self.heatmap.shape[:2]
        
        # If dimensions don't match, resize heatmap to match frame
        if frame_h != heatmap_h or frame_w != heatmap_w:
            heatmap_display = cv2.resize(self.heatmap, (frame_w, frame_h))
        else:
            heatmap_display = self.heatmap.copy()
        
        heatmap_min = heatmap_display.min()
        heatmap_max = heatmap_display.max()

        if heatmap_max > heatmap_min:
            heatmap_display = (heatmap_display - heatmap_min) / (heatmap_max - heatmap_min) * 255
        else:
            heatmap_display = np.zeros_like(heatmap_display)

        heatmap_display = heatmap_display.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_display, cv2.COLORMAP_JET)
        
        # Ensure heatmap_color has same dimensions as frame
        if heatmap_color.shape != frame.shape:
            heatmap_color = cv2.resize(heatmap_color, (frame_w, frame_h))

        # Safe addWeighted with error handling
        try:
            return cv2.addWeighted(frame, 0.7, heatmap_color, 0.5, 0)
        except cv2.error as e:
            # If still fails, return original frame
            print(f"Heatmap overlay error: {e}")
            return frame.copy()

    # ✅ REQUIRED BY Predict_main.py
    def reset(self):
        self.heatmap = None
        self.prev_landmarks = None

    # ✅ REQUIRED BY Predict_main.py (H key)
    def toggle(self):
        self.show_heatmap = not self.show_heatmap
        return self.show_heatmap
