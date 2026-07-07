import cv2
import numpy as np
import os

VEHICLE_DIMS = {
  'car': 1.8,
  'truck': 2.6
}

def detect_vehicles(image_path, background_model):
  frame = cv2.imread(image_path)
  roi = frame[100:255, 0:480]

  fg_mask = background_model.apply(roi)

  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
  fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

  contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  vehicles = []

  for contour in contours:
    area = cv2.contourArea(contour)

    if area < 200 or area > 15000:
      continue

    x, y, w, h = cv2.boundingRect(contour)
    aspect = w / (h + 1e-5)
    
    if aspect > 2.0 and area > 2000:
      vehicle_type = 'truck'
    else:
      vehicle_type = 'car'

    real_width = VEHICLE_DIMS[vehicle_type]
    meters_per_pixel = real_width / w

    bottom_center_x = x + w // 2 
    bottom_center_y = y + h + 100

    vehicles.append({
      'type': vehicle_type,
      'pixel_width': w,
      'real_width': real_width,
      'meters_per_pixel': meters_per_pixel,
      'bottom_center_x': bottom_center_x,
      'bottom_center_y': bottom_center_y
    })
  
  return vehicles

frames_dir = '../frames'
all_vehicles = []

bg_model = cv2.createBackgroundSubtractorMOG2(
  history=300,
  varThreshold=40,
  detectShadows=False
)

for filename in sorted(os.listdir(frames_dir)):
  if not filename.endswith('.jpg'):
    continue

  path = os.path.join(frames_dir, filename)
  vehicles = detect_vehicles(path, bg_model)
  all_vehicles.extend(vehicles)

  if vehicles:
    print(f'{filename}: {len(vehicles)} vehicles detected')
    for v in vehicles:
      print(f"  {v['type']} — pixel width: {v['pixel_width']}px, meters/pixel: {v['meters_per_pixel']:.4f}, ground pixel: ({v['bottom_center_x']}, {v['bottom_center_y']})")
  
print(f'\nTotal vehicles detected: {len(all_vehicles)}')

if all_vehicles:
  print(f'Avg meters per pixel: {np.mean([v['meters_per_pixel'] for v in all_vehicles]):.4f}')
