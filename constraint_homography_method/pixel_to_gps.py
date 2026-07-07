import cv2
import numpy as np

CAMERA_LAT = 41.99968
CAMERA_LON = -91.67751

def pixel_to_gps(u, v, H):
  pixel_coord = np.array([[[u, v]]], dtype=np.float32)
  result = cv2.perspectiveTransform(pixel_coord, H)

  X_m = float(result[0][0][0])
  Y_m = float(result[0][0][1])

  delta_lat = Y_m / 111320.0
  delta_lon = X_m / (111320.0 * np.cos(np.radians(CAMERA_LAT)))

  return CAMERA_LAT + delta_lat, CAMERA_LON + delta_lon

H = np.load('H_matrix_CRTV12.npy')

test_pixels = [
  (240, 200, 'Center frame, mid distance'),
  (240, 240, 'Center frame, near camera'),
  (150, 210, 'Left lane, mid distance'),
  (350, 210, 'Right lane, mid distance')
]

for u, v, label in test_pixels:
  lat, lon = pixel_to_gps(u, v, H)
  print(label)
  print(f'Pixel ({u}, {v}) -> {lat:.6f}, {lon:.6f}')
  print(f'Google Maps: https://maps.google.com/?q={lat},{lon}')
  print()