import cv2
import os

os.makedirs('../frames', exist_ok=True)

video_files = [
  "video_files/video_CRTV12_113631411000.mp4",
  "video_files/video_CRTV12_115248035000.mp4",
  "video_files/video_CRTV12_115652004000.mp4",
  "video_files/video_CRTV12_115702341000.mp4",
  "video_files/video_CRTV12_142656995000.mp4",
  "video_files/video_CRTV12_145145179000.mp4",
  "video_files/video_CRTV12_145233559000.mp4",
  "video_files/video_CRTV12_145523685000.mp4",
  "video_files/video_CRTV12_150126797000.mp4",
  "video_files/video_CRTV12_165821380000.mp4"
]

saved_frames = 0

for video_file in video_files:
  cap = cv2.VideoCapture(video_file)
  fps = cap.get(cv2.CAP_PROP_FPS)

  frame_idx = 0
  
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    if frame_idx % int(fps * 2) == 0:
      cv2.imwrite(f'../frames/frame_{saved_frames}.jpg', frame)
      saved_frames += 1
    frame_idx += 1
  
  cap.release()
  print(f"Done: {video_file}")

print(f"Total frames saved: {saved_frames}")