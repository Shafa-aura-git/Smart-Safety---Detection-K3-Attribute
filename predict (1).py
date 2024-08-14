from ultralytics import YOLO
import cv2
import time 

model = YOLO("smarfa.pt")

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise("No camera")

while True:
    ret, image = cam.read()
    if not ret:
        break

    time_mulai = time.time()
    result = model.predict(image, show= True)

    print("waktu", time.time()-time_mulai)
    _key = cv2.waitKey(1)
    if _key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()