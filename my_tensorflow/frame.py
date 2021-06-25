import cv2
import os


cap = cv2.VideoCapture('/home/jacob_chiachuyou/squat_roundedback_chuyou.mp4')
success,image = cap.read()
count = 0
while success:
  cv2.imwrite(os.path.join(os.getcwd() + "/test/", "frame%d.jpg" % count), image)
  success,image = cap.read()
  print('Read a new frame: ', success)
  count += 10 # i.e. at 30 fps, this advances one second
  cap.set(1, count)  
