import cv2
import numpy as np
photo = cv2.imread('1.jpg')  # SUCCESS: 1
photoGray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
photoEnhance = cv2.equalizeHist(photoGray)
photoBlur = cv2.GaussianBlur(photoEnhance, (3, 3), 0)
ret, photo2 = cv2.threshold(photoBlur, 197, 255, cv2.THRESH_TOZERO_INV) # 197
ret2, photo3 = cv2.threshold(photo2, 193, 255, cv2.THRESH_BINARY) # 193
kernel = np.ones((3, 3), np.uint8)
pictureO = cv2.morphologyEx(photo3, cv2.MORPH_CLOSE, kernel)
lines = cv2.HoughLinesP(pictureO, 1, np.pi / 180, 100, 100, 90, 4)

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(photo, (x2, y2), (x1, y1), (0, 0, 255), 2)

cv2.imshow('final', photo)
cv2.waitKey(0)
cv2.destroyAllWindows()