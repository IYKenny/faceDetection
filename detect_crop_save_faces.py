import cv2
img = cv2.imread("/home/iykenny/Desktop/faceDetection/couple.jpg")

image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier("/home/iykenny/Downloads/haarcascade_frontalface_alt.xml")

detected_faces = face_cascade.detectMultiScale(image_gray, 1.1, 4)
print('Number of detected faces:', len(detected_faces))

if len(detected_faces) > 0:
    for i, (x, y, w, h) in enumerate(detected_faces):
        # To draw a rectangle in a face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        face = img[y:y + h, x:x + w]
        cv2.imshow(f"Face {i}", face)
        cv2.imwrite(f'face{i}.jpg', face)
        print(f"face{i}.jpg is saved")

cv2.imshow("Picture", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


