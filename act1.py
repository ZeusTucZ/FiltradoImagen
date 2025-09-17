import cv2
import numpy as np
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt

img = cv2.imread('placa_q.jpg')

h, w = img.shape[:2]
band = int(h * 0.50)
y1 = (h - band) // 2
roi = img[y1:y1+band, :]

gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
T = 110
mask = cv2.inRange(gray, 0, T)

k3 = np.ones((3,3), np.uint8)
mask = cv2.erode(mask, k3, iterations=1)
mask = cv2.dilate(mask, k3, iterations=1)

k5 = np.ones((5,5), np.uint8)
mask = cv2.dilate(mask, k5, iterations=1)
mask = cv2.erode(mask, k5, iterations=1)

mask = cv2.dilate(mask, k3, iterations=1)

white = np.full_like(roi, 255)
text_rgb = cv2.bitwise_and(roi, roi, mask=mask)
inv = cv2.bitwise_not(mask)
bg = cv2.bitwise_and(white, white, mask=inv)
final = cv2.add(text_rgb, bg)
final_gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)

config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
text = pytesseract.image_to_string(final_gray, config=config).strip()
print("placa_q.jpg →", text)

plt.figure(figsize=(12,3))
plt.subplot(1,4,1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title('Original'); plt.axis('off')
plt.subplot(1,4,2); plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)); plt.title('ROI'); plt.axis('off')
plt.subplot(1,4,3); plt.imshow(mask, cmap='gray'); plt.title('Máscara'); plt.axis('off')
plt.subplot(1,4,4); plt.imshow(final_gray, cmap='gray'); plt.title('Procesada'); plt.axis('off')
plt.suptitle('placa_q.jpg')
plt.tight_layout()
plt.show()


img = cv2.imread('placa_4.jpg')

h, w = img.shape[:2]
band = int(h * 0.50)
y1 = (h - band) // 2
roi = img[y1:y1+band, :]

gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
T = 110
mask = cv2.inRange(gray, 0, T)

k3 = np.ones((3,3), np.uint8)
mask = cv2.erode(mask, k3, iterations=1)
mask = cv2.dilate(mask, k3, iterations=1)

k5 = np.ones((5,5), np.uint8)
mask = cv2.dilate(mask, k5, iterations=1)
mask = cv2.erode(mask, k5, iterations=1)

mask = cv2.dilate(mask, k3, iterations=1)

white = np.full_like(roi, 255)
text_rgb = cv2.bitwise_and(roi, roi, mask=mask)
inv = cv2.bitwise_not(mask)
bg = cv2.bitwise_and(white, white, mask=inv)
final = cv2.add(text_rgb, bg)
final_gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)

config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
text = pytesseract.image_to_string(final_gray, config=config).strip()
print("placa_4.jpg →", text)

# Mostrar con Matplotlib (imagen 2)
plt.figure(figsize=(12,3))
plt.subplot(1,4,1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title('Original'); plt.axis('off')
plt.subplot(1,4,2); plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)); plt.title('ROI'); plt.axis('off')
plt.subplot(1,4,3); plt.imshow(mask, cmap='gray'); plt.title('Máscara'); plt.axis('off')
plt.subplot(1,4,4); plt.imshow(final_gray, cmap='gray'); plt.title('Procesada'); plt.axis('off')
plt.suptitle('placa_4.jpg')
plt.tight_layout()
plt.show()
