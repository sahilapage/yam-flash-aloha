"""zoom.py IMG X0 Y0 W H OUT: 3x crop with a 10 px grid labelled in full-image pixels."""
import sys, cv2
img = cv2.imread(sys.argv[1]); x0, y0, w, h = map(int, sys.argv[2:6]); s = 3
c = cv2.resize(img[y0:y0 + h, x0:x0 + w], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
for i in range(0, w, 10):
    cv2.line(c, (i * s, 0), (i * s, h * s), (0, 0, 255) if (x0 + i) % 50 == 0 else (0, 200, 255), 1)
    if (x0 + i) % 50 == 0: cv2.putText(c, str(x0 + i), (i * s + 2, 14), 0, 0.5, (0, 0, 255), 1)
for j in range(0, h, 10):
    cv2.line(c, (0, j * s), (w * s, j * s), (0, 0, 255) if (y0 + j) % 50 == 0 else (0, 200, 255), 1)
    if (y0 + j) % 50 == 0: cv2.putText(c, str(y0 + j), (2, j * s - 2), 0, 0.5, (0, 0, 255), 1)
cv2.imwrite(sys.argv[6], c)
