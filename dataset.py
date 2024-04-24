import tkinter as tk
import cv2
import os
import numpy as np
from PIL import Image

def start_capture(name):
    path = "./data/" + name
    num_of_images = 0
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    try:
        os.makedirs(path)
    except:
        print('Directory Already Created')
    vid = cv2.VideoCapture(0)
    while True:

        ret, img = vid.read()
        new_img = None
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = detector.detectMultiScale(image=grayimg, scaleFactor=1.1, minNeighbors=5)
        for x, y, w, h in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
            cv2.putText(img, "Face Detected", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
            cv2.putText(img, str(str(num_of_images) + " images captured"), (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255))
            new_img = img[y:y + h, x:x + w]
        cv2.imshow("Face Detection", img)
        key = cv2.waitKey(1) & 0xFF

        try:
            cv2.imwrite(str(path + "/" + str(num_of_images) + name + ".jpg"), new_img)
            num_of_images += 1
        except:
            pass
        if key == ord("q") or key == 27 or num_of_images > 300:  # take 300 frames
            break
    cv2.destroyAllWindows()
    return num_of_images

def train_classifier(name):
    # Read all the images in custom data-set
    path = os.path.join(os.getcwd() + "/data/" + name + "/")

    faces = []
    ids = []
    labels = []
    pictures = {}

    # Store images in a numpy format and ids of the user on the same index in imageNp and id lists

    for root, dirs, files in os.walk(path):
        pictures = files

    for pic in pictures:
        imgpath = path + pic
        img = Image.open(imgpath).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(pic.split(name)[0])
        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    # Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write(name + "_classifier.xml")

def capture_images():
    name = entry.get()
    num_images = start_capture(name)
    print("Captured", num_images, "images for", name)

def train_classifier_button():
    name = entry.get()
    train_classifier(name)
    print("Classifier trained for", name)

# Create the main window
root = tk.Tk()
root.title("Face Recognition System")

# Create a label and an entry widget for the variable name
label = tk.Label(root, text="Enter Name:")
label.grid(row=0, column=0)

entry = tk.Entry(root)
entry.grid(row=0, column=1)

# Create buttons to capture images and train classifier
capture_button = tk.Button(root, text="Start Capture", command=capture_images)
capture_button.grid(row=1, column=0)

train_button = tk.Button(root, text="Train Classifier", command=train_classifier_button)
train_button.grid(row=1, column=1)

# Run the tkinter event loop
root.mainloop()
