import cv2

print("OpenCV version:", cv2.__version__)

# Try to load an image
img = cv2.imread("data/test.jpg")

if img is None:
    print("Image not found, but OpenCV is working!")
else:
    print("Image loaded successfully!")
    cv2.imshow("Test Image", img)
    cv2.waitKey(0)
