import cv2
import numpy as np
import os

def detect_lane_markings(image_path):
  frame = cv2.imread(image_path)
  roi = frame[100:255, 0:480]

  grayscale = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  contrast = clahe.apply(grayscale)

  edges = cv2.Canny(contrast, 40, 120)

  lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi/180,
    threshold=25,
    minLineLength=15,
    maxLineGap=5
  )

  if lines is None:
    return []

  lane_segments = []
  
  for line in lines:
    x1, y1, x2, y2 = line[0]

    dx = abs(x2-x1)
    dy = abs(y2-y1)

    if dy < 5 or dx / (dy + 1e-5) > 0.5:
      continue

    pixel_length = np.sqrt(dx**2 + dy**2)
    meters_per_pixel = 3.0 / pixel_length

    lane_segments.append({
      'pixel_length': pixel_length,
      'meters_per_pixel': meters_per_pixel,
      'x1': x1, 'y1': y1, 'x2': x1, 'y2': y2
    })
  
  return lane_segments

frames_dir = '../frames'
all_segments = []

for filename in sorted(os.listdir(frames_dir)):
  if not filename.endswith('.jpg'):
    continue

  path = os.path.join(frames_dir, filename)
  segments = detect_lane_markings(path)
  all_segments.extend(segments)

  if segments:
    print(f'{filename}: {len(segments)} lane segments detected\nAvg meters per pixel: {np.mean([s['meters_per_pixel'] for s in segments]):.4f}')
  
print(f'\nTotal lane segments detected: {len(all_segments)}')

if all_segments:
  print(f'Avg meters per pixel: {np.mean([s['meters_per_pixel'] for s in all_segments]):.4f}')
