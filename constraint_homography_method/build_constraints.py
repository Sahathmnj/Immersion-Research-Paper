import cv2
import numpy as np
import os
from detect_lanes import detect_lane_markings
from detect_vehicles import detect_vehicles

def build_constraints(frames_dir):
  pixel_points = []
  real_world_points = []

  bg_model = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=40,
    detectShadows=False
  )

  for filename in sorted(os.listdir(frames_dir)):
    if not filename.endswith('.jpg'):
      continue

    path = os.path.join(frames_dir, filename)

    lane_segments = detect_lane_markings(path)
    
    for segment in lane_segments:
      u = (segment['x1'] + segment['x2']) // 2
      v = max(segment['y1'], segment['y2']) + 100

      norm_y = (v - 100) / ((255 - 100))
      forward_m = 60.0 - norm_y * 55.0

      frame_center_x = 240
      lateral_m = (u - frame_center_x) * segment['meters_per_pixel']

      pixel_points.append([u, v])
      real_world_points.append([lateral_m, forward_m])

    vehicles = detect_vehicles(path, bg_model)

    for vehicle in vehicles:
      u = vehicle['bottom_center_x']
      v = vehicle['bottom_center_y']

      norm_y = (v - 100) / ((255 - 100))
      forward_m = 60.0 - norm_y * 55.0

      frame_center_x = 240
      lateral_m = (u - frame_center_x) * vehicle['meters_per_pixel']

      pixel_points.append([u, v])
      real_world_points.append([lateral_m, forward_m])

  pixel_points = np.array(pixel_points, dtype=np.float32)
  real_world_points = np.array(real_world_points, dtype=np.float32)

  print(f'Total constraints built: {len(pixel_points)}')
  print(f'From lane markings and vehicles')

  return pixel_points, real_world_points

pixel_points, real_world_points = build_constraints('../frames')

print('\nSample constraints pairs:')
for i in range(min(5, len(pixel_points))):
  print(f'Pixel: ({pixel_points[i][0]:.0f}, {pixel_points[i][1]:.0f}) -> Real World: ({real_world_points[i][0]:.2f}m, {real_world_points[i][1]:.2f}m)')