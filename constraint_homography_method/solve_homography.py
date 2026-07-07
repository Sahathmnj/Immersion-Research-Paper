import cv2
import numpy as np
import json
import os
from build_constraints import build_constraints

pixel_points, real_world_points = build_constraints('../frames')

H, mask = cv2.findHomography(
  pixel_points,
  real_world_points,
  cv2.RANSAC,
  ransacReprojThreshold=3.0
)

inliers = mask.sum()
total = len(mask)

print(f'H matrix computed successfully')
print(f'Inliers: {inliers}/{total} ({inliers / total:.0%})')
print(f'\nH matrix:')
print(H)

np.save('H_matrix_CRTV12.npy', H)
print('\nH matrix saved to H_matrix_CRTV12.npy')
